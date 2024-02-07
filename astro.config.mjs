import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from "@astrojs/tailwind";

import vercel from "@astrojs/vercel/serverless";

// https://astro.build/config
export default defineConfig({
  site: "https://beebekisme.vercel.app/",
  base: "/",
  integrations: [mdx(), sitemap(), tailwind()],
  markdown: {
    shikiConfig: {
      theme: 'monokai',
      langs: ['python'],
      wrap: true,
      transformers: []
    }
  },
  output: "hybrid",
  adapter: vercel()
});