// Define global types for the app
interface AppEnv {
  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
}

declare global {
  interface Window {
    ENV: AppEnv;
  }

  // Allow ENV to be set on globalThis for server-side rendering.
  // Ambient globals require `var` for TypeScript declaration merging.
  // deno-lint-ignore no-var
  var ENV: AppEnv; // eslint-disable-line no-var
}

export {};
