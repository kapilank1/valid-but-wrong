import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The repo (and the home dir above it) hold other lockfiles; pin tracing to this app.
  outputFileTracingRoot: process.cwd(),
  // Lint runs separately; a missing ESLint install should never block a demo build.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
