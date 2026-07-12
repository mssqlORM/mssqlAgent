import { embedder } from 'genkit/plugin';
import http from 'http';
import https from 'https';
interface EmbedConfig {
  endpoint: string;
  apiKey: string;
  model: string;
}

function getConfig(): EmbedConfig | null {
  const endpoint = process.env.EMBEDDING_ENDPOINT || process.env.LLM_ENDPOINT || '';
  const apiKey = process.env.EMBEDDING_API_KEY || process.env.LLM_API_KEY || '';
  const model = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
  if (!endpoint || !apiKey) return null;
  return { endpoint, apiKey, model };
}

function fetchJson(url: string, options: any, timeoutMs = 30000): Promise<string> {
  return new Promise((resolve, reject) => {
    const lib = /^https:/.test(url) ? https : http;
    const req = lib.request(url, options, (res) => {
      let data = '';
      res.on('data', (chunk: string) => (data += chunk));
      res.on('end', () => resolve(data));
    });
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => { req.destroy(); reject(new Error('Embedding request timed out')); });
    req.write(options.body || '');
    req.end();
  });
}

async function callEmbeddingEndpoint(texts: string[], config: EmbedConfig): Promise<number[][]> {
  const body = JSON.stringify({ model: config.model, input: texts });
  const response = await fetchJson(config.endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${config.apiKey}`,
    },
    body,
  });
  const parsed = JSON.parse(response);

  // OpenAI format: { data: [{ embedding: [...] }, ...] }
  if (parsed.data && Array.isArray(parsed.data)) {
    return parsed.data.map((d: any) => d.embedding);
  }
  // Cohere / generic format: { embeddings: [[...], ...] }
  if (parsed.embeddings && Array.isArray(parsed.embeddings)) {
    return parsed.embeddings;
  }
  throw new Error('Unknown embedding response format');
}

function dummyEmbed(text: string): number[] {
  // Deterministic hash-based fake vector for offline fallback (384 dims)
  const dim = 384;
  const vec = new Array(dim).fill(0);
  for (let i = 0; i < text.length; i++) {
    vec[i % dim] = (vec[i % dim] + text.charCodeAt(i)) % 1000 / 1000;
  }
  // L2 normalize
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  return vec.map((v) => v / norm);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const customEmbedder: any = embedder(
  { name: 'custom-embedder', info: { label: 'Custom Embedder', dimensions: 384 } },
  async (input: any) => {
    const config = getConfig();
    const docs = input.input || input.embeddings || [];
    const texts = docs.map((d: any) => {
      const content = d.content || (d.text !== undefined ? d.text : d);
      return typeof content === 'string' ? content : JSON.stringify(content);
    });

    if (config && texts.length > 0) {
      try {
        const vectors = await callEmbeddingEndpoint(texts, config);
        return { embeddings: vectors.map((embedding: number[]) => ({ embedding })) };
      } catch (err: any) {
        console.warn(`[embedder] endpoint failed: ${err.message}, using dummy embeddings`);
      }
    }

    // Fallback: deterministic dummy embeddings (for offline development)
    return { embeddings: texts.map((t: string) => ({ embedding: dummyEmbed(t) })) };
  }
);
