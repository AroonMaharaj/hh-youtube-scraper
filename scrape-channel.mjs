/**
 * YouTube Channel Scraper + Pinecone Ingestion
 *
 * Scrapes long-form videos from @hungryhorsepoker, gets transcripts,
 * embeds them with Google Gemini, and upserts to Pinecone.
 * Skips videos already in Pinecone (tracked via manifest vectors).
 *
 * Usage: node scripts/scrape-channel.mjs [--dry-run]
 */
import 'dotenv/config';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from '@google/genai';
import { fetchTranscript } from 'youtube-transcript/dist/youtube-transcript.esm.js';
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const blocklist = JSON.parse(readFileSync(join(__dirname, 'blocklist.json'), 'utf-8'));

function isBlocked(title) {
  const lower = title.toLowerCase();
  for (const pattern of blocklist.blockedTitlePatterns) {
    if (lower.includes(pattern.toLowerCase())) return true;
  }
  for (const blocked of blocklist.blockedTitles) {
    if (lower.includes(blocked.toLowerCase())) return true;
  }
  return false;
}

// ── Config ───────────────────────────────────────────────────────────────
const CHANNEL_HANDLE = '@hungryhorsepoker';
const INDEX_NAME = 'mark-goone-kb';
const NAMESPACE = 'youtube';
const CHUNK_SIZE = 1500;
const CHUNK_OVERLAP = 200;
const SHORT_VIDEO_THRESHOLD_SEC = 300; // 5 min — skip all shorts and short clips
const DRY_RUN = process.argv.includes('--dry-run');

// ── Clients ──────────────────────────────────────────────────────────────
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(INDEX_NAME);
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ── YouTube Fetching ─────────────────────────────────────────────────────

function parseDuration(text) {
  if (!text) return 0;
  const parts = text.split(':').map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return parts[0] || 0;
}

/** Fetch the upload date from a YouTube video page */
async function fetchUploadDate(videoId) {
  try {
    const res = await fetch(`https://www.youtube.com/watch?v=${videoId}`, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
    });
    const html = await res.text();
    const match = html.match(/"dateText":\{"simpleText":"([^"]+)"\}/)
      || html.match(/"publishDate":"([^"]+)"/)
      || html.match(/"uploadDate":"([^"]+)"/)
      || html.match(/<meta itemprop="datePublished" content="([^"]+)"/);
    return match?.[1] || null;
  } catch {
    return null;
  }
}

/** Scrape the channel /videos page for video metadata */
async function scrapeChannelPage() {
  console.log(`Fetching ${CHANNEL_HANDLE}/videos ...`);
  const res = await fetch(`https://www.youtube.com/${CHANNEL_HANDLE}/videos`, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Accept-Language': 'en-US,en;q=0.9',
    },
  });
  const html = await res.text();
  const videos = [];

  const dataMatch = html.match(/var ytInitialData = ({.*?});\s*<\/script>/s);
  if (dataMatch) {
    try {
      const data = JSON.parse(dataMatch[1]);
      const tabs = data?.contents?.twoColumnBrowseResultsRenderer?.tabs || [];
      for (const tab of tabs) {
        const contents = tab?.tabRenderer?.content?.richGridRenderer?.contents || [];
        for (const item of contents) {
          const video = item?.richItemRenderer?.content?.videoRenderer;
          if (video?.videoId) {
            videos.push({
              id: video.videoId,
              title: video.title?.runs?.[0]?.text || 'Untitled',
              duration: video.lengthText?.simpleText || '',
              durationSec: parseDuration(video.lengthText?.simpleText || ''),
            });
          }
        }
      }
    } catch (e) {
      console.error('Failed to parse ytInitialData:', e.message);
    }
  }

  return videos;
}

/** Get channel ID for RSS feed */
async function getChannelId() {
  const res = await fetch(`https://www.youtube.com/${CHANNEL_HANDLE}`, {
    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
  });
  const html = await res.text();
  const match = html.match(/"channelId":"([a-zA-Z0-9_-]+)"/) ||
                html.match(/"externalId":"([a-zA-Z0-9_-]+)"/);
  return match?.[1] || null;
}

/** Get recent video IDs from RSS (catches new uploads between page scrapes) */
async function getRSSVideoIds(channelId) {
  const res = await fetch(`https://www.youtube.com/feeds/videos.xml?channel_id=${channelId}`);
  const xml = await res.text();
  const ids = [];
  const entries = xml.split('<entry>').slice(1);
  for (const entry of entries) {
    const idMatch = entry.match(/<yt:videoId>([^<]+)<\/yt:videoId>/);
    if (idMatch) ids.push(idMatch[1]);
  }
  return ids;
}

// ── Pinecone Helpers ─────────────────────────────────────────────────────

/** Get set of video IDs already indexed in Pinecone */
async function getExistingVideoIds() {
  const existing = new Set();
  try {
    let paginationToken = undefined;
    do {
      const page = await index.namespace(NAMESPACE).listPaginated({
        prefix: 'manifest_',
        ...(paginationToken ? { paginationToken } : {}),
      });
      if (page.vectors) {
        for (const v of page.vectors) {
          existing.add(v.id.replace('manifest_', ''));
        }
      }
      paginationToken = page.pagination?.next;
    } while (paginationToken);
  } catch (e) {
    console.log('Could not list existing vectors:', e.message);
  }
  return existing;
}

// ── Text Processing ──────────────────────────────────────────────────────

function chunkText(text) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + CHUNK_SIZE, text.length);
    chunks.push(text.slice(start, end));
    start += CHUNK_SIZE - CHUNK_OVERLAP;
  }
  return chunks;
}

async function getTranscript(videoId) {
  try {
    const segments = await fetchTranscript(videoId);
    return segments.map((s) => s.text).join(' ');
  } catch (e) {
    return null;
  }
}

// ── Embedding ────────────────────────────────────────────────────────────

/** Embed a single text with retry + exponential backoff on rate limits */
async function embedWithRetry(text, maxRetries = 5) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const result = await ai.models.embedContent({
        model: 'gemini-embedding-001',
        contents: text,
        config: { taskType: 'RETRIEVAL_DOCUMENT' },
      });
      return result.embeddings[0].values;
    } catch (e) {
      const msg = typeof e === 'object' ? JSON.stringify(e) : String(e);
      const isRateLimit = msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED');

      if (isRateLimit && attempt < maxRetries) {
        const waitSec = Math.pow(2, attempt) * 15; // 15s, 30s, 60s, 120s, 240s
        console.log(`    Rate limited, waiting ${waitSec}s (attempt ${attempt + 1}/${maxRetries})...`);
        await sleep(waitSec * 1000);
      } else {
        throw e;
      }
    }
  }
}

async function embedChunks(chunks) {
  const embeddings = [];
  for (let i = 0; i < chunks.length; i++) {
    const vector = await embedWithRetry(chunks[i]);
    embeddings.push(vector);

    // Rate limit: pause every 5 embeddings
    if ((i + 1) % 5 === 0) {
      await sleep(300);
    }

    // Progress indicator every 10 chunks
    if ((i + 1) % 10 === 0) {
      console.log(`    Embedded ${i + 1}/${chunks.length} chunks`);
    }
  }
  return embeddings;
}

// ── Pinecone Upsert ──────────────────────────────────────────────────────

async function upsertVideo(video, chunks, embeddings) {
  if (!embeddings.length || embeddings.length !== chunks.length) {
    throw new Error(`Embedding count mismatch: ${embeddings.length} embeddings for ${chunks.length} chunks`);
  }

  const vectors = [];

  for (let i = 0; i < chunks.length; i++) {
    if (!embeddings[i]) continue;
    vectors.push({
      id: `${video.id}_chunk${i}`,
      values: embeddings[i],
      metadata: {
        videoId: video.id,
        title: video.title,
        duration: video.duration,
        uploadDate: video.uploadDate || '',
        chunkIndex: i,
        totalChunks: chunks.length,
        text: chunks[i].slice(0, 3600),
        source: 'youtube',
      },
    });
  }

  // Manifest vector for tracking
  vectors.push({
    id: `manifest_${video.id}`,
    values: embeddings[0],
    metadata: {
      videoId: video.id,
      title: video.title,
      duration: video.duration,
      uploadDate: video.uploadDate || '',
      isManifest: true,
      totalChunks: chunks.length,
      source: 'youtube',
      scrapedAt: new Date().toISOString(),
    },
  });

  if (vectors.length === 0) {
    throw new Error('No vectors to upsert');
  }

  // Upsert in batches of 100
  for (let i = 0; i < vectors.length; i += 100) {
    const batch = vectors.slice(i, i + 100);
    if (batch.length > 0) {
      await index.namespace(NAMESPACE).upsert({ records: batch });
    }
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Main ─────────────────────────────────────────────────────────────────

async function main() {
  const startTime = Date.now();
  console.log('=== YouTube → Pinecone Ingestion ===');
  console.log(`Channel: ${CHANNEL_HANDLE}`);
  console.log(`Index: ${INDEX_NAME} / ${NAMESPACE}`);
  console.log(`Mode: ${DRY_RUN ? 'DRY RUN' : 'LIVE'}\n`);

  // 1. Get already-indexed video IDs
  const existingIds = await getExistingVideoIds();
  console.log(`Already indexed: ${existingIds.size} videos`);

  // 2. Fetch videos from channel page + RSS
  const pageVideos = await scrapeChannelPage();
  console.log(`Channel page: ${pageVideos.length} videos`);

  const channelId = await getChannelId();
  const rssIds = channelId ? await getRSSVideoIds(channelId) : [];
  console.log(`RSS feed: ${rssIds.length} video IDs`);

  // Merge: page videos (with metadata) + RSS-only IDs
  const videoMap = new Map();
  for (const v of pageVideos) videoMap.set(v.id, v);
  for (const id of rssIds) {
    if (!videoMap.has(id)) {
      videoMap.set(id, { id, title: '', duration: '', durationSec: 0 });
    }
  }

  // Filter to long-form only (skip videos under threshold, but keep unknown-duration from RSS)
  // Also filter out blocked videos (vlog episodes, collabs, non-strategy content)
  const allVideos = [...videoMap.values()].filter(
    (v) => (v.durationSec === 0 || v.durationSec >= SHORT_VIDEO_THRESHOLD_SEC) && !isBlocked(v.title)
  );
  console.log(`Total unique long-form candidates: ${allVideos.length} (after blocklist filter)`);

  // 3. Filter out already-scraped
  const newVideos = allVideos.filter((v) => !existingIds.has(v.id));
  console.log(`New videos to process: ${newVideos.length}\n`);

  if (newVideos.length === 0) {
    console.log('No new videos. Done!');
    return;
  }

  // 4. Process each video
  let processed = 0;
  let skipped = 0;

  for (let idx = 0; idx < newVideos.length; idx++) {
    const video = newVideos[idx];
    console.log(`[${idx + 1}/${newVideos.length}] ${video.title || video.id} (${video.duration || 'unknown duration'})`);

    // Get transcript
    const transcript = await getTranscript(video.id);
    if (!transcript) {
      console.log('  SKIP — no transcript available');
      skipped++;
      continue;
    }

    // If duration was unknown, check transcript length to filter shorts
    if (video.durationSec === 0 && transcript.length < 5000) {
      console.log(`  SKIP — transcript too short (${transcript.length} chars, likely a short/clip)`);
      skipped++;
      continue;
    }

    // Get title via oEmbed if missing
    if (!video.title) {
      try {
        const oembed = await fetch(`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${video.id}&format=json`);
        const data = await oembed.json();
        video.title = data.title || video.id;
      } catch {
        video.title = video.id;
      }
    }

    // Check blocklist after title is resolved
    if (isBlocked(video.title)) {
      console.log(`  SKIP — blocked by blocklist: "${video.title}"`);
      skipped++;
      continue;
    }

    // Fetch upload date
    const uploadDate = await fetchUploadDate(video.id);
    if (uploadDate) {
      video.uploadDate = uploadDate;
      console.log(`  Upload date: ${uploadDate}`);
    }

    console.log(`  Transcript: ${transcript.length} chars`);
    const chunks = chunkText(transcript);
    console.log(`  Chunks: ${chunks.length}`);

    if (DRY_RUN) {
      console.log('  [DRY RUN] Would embed and upsert');
      processed++;
      continue;
    }

    // Embed
    console.log('  Embedding...');
    try {
      const embeddings = await embedChunks(chunks);

      // Upsert
      console.log('  Upserting to Pinecone...');
      await upsertVideo(video, chunks, embeddings);
      console.log('  Done!');
      processed++;
    } catch (e) {
      console.error(`  ERROR: ${e.message}`);
      skipped++;
    }

    // Rate limit between videos
    await sleep(1500);
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n=== Complete (${elapsed}s) ===`);
  console.log(`Processed: ${processed} | Skipped: ${skipped} | Already indexed: ${existingIds.size}`);

  if (!DRY_RUN) {
    const stats = await index.describeIndexStats();
    console.log(`Pinecone total records: ${stats.totalRecordCount}`);
  }
}

main().catch((e) => {
  console.error('Fatal error:', e);
  process.exit(1);
});
