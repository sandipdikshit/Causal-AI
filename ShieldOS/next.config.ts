import type { NextConfig } from "next";

const githubDemoBase = process.env.GITHUB_DEMO
  ? "/sandipdikshit/Causal-AI/main/ShieldOS/demo"
  : "";

const nextConfig: NextConfig = {
  output: "export",
  basePath: githubDemoBase,
  assetPrefix: githubDemoBase,
};

export default nextConfig;
