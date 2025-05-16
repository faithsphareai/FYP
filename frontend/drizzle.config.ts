import type { Config } from 'drizzle-kit';

export default {
  schema: './db/schema.js',
  out: './drizzle',
  dialect: 'sqlite',
  driver: 'expo',
} as Config;