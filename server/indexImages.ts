/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable dot-notation */
import * as dotenv from "dotenv";
import { Pinecone } from '@pinecone-database/pinecone';
import type { PineconeRecord } from '@pinecone-database/pinecone';
import { embedder } from "./embeddings.ts";
import { getEnv, listFiles } from "./utils/util.ts";
import { chunkedUpsert } from "./utils/chunkedUpsert.ts";

dotenv.config();

// Index setup
const indexName = getEnv("PINECONE_INDEX");
const apiKey = getEnv("PINECONE_API_KEY");
const environment = getEnv("PINECONE_ENVIRONMENT");

const pinecone = new Pinecone({
  apiKey, 
  environment,
});

function* chunkArray<T>(array: T[], chunkSize: number): Generator<T[]> {
  for (let i = 0; i < array.length; i += chunkSize) {
    yield array.slice(i, i + chunkSize);
  }
}

async function embedAndUpsert({ imagePaths, chunkSize }: { imagePaths: string[], chunkSize: number }) {
  // Chunk the image paths into batches of size chunkSize
  const chunkGenerator = chunkArray(imagePaths, chunkSize);

  // Get the index
  const index = pinecone.index(indexName);

  // Embed each batch and upsert the embeddings into the index
  for await (const imagePaths of chunkGenerator) {
    await embedder.embedBatch(imagePaths, chunkSize, async (embeddings: PineconeRecord[]) => {
      await chunkedUpsert(index, embeddings, "default");
    });
  }
}

const indexImages = async () => {
  try {
    // Create the index if it doesn't already exist
    const indexList = await pinecone.listIndexes();
    for (const index of indexList) {
      if (index.name === indexName) {
        console.log(`Index ${indexName} already exists`);
       // return;
      }
    }
    //
    // if (indexList.indexOf({ name: indexName }) === -1) {
    //  await pinecone.createIndex({ name: indexName, dimension: 512, waitUntilReady: true });
    // }

    await embedder.init("Xenova/clip-vit-base-patch32");
    const imagePaths = await listFiles("./data");
    await embedAndUpsert({ imagePaths, chunkSize: 100 });
    return;

  } catch (error) {
    console.error(error);
    throw error;
  }
};

export {
  indexImages
};

