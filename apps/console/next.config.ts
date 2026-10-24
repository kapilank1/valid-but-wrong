import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Lint runs separately; a missing ESLint install should never block a demo build.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
