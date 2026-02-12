import { createClient } from "@supabase/supabase-js";

const supabaseUrl = globalThis.ENV.SUPABASE_URL || "";
const supabaseAnonKey = globalThis.ENV.SUPABASE_ANON_KEY || "";

if (!supabaseUrl || !supabaseAnonKey) {
  console.error("Missing Supabase environment variables");
}

// Create a single supabase client for the entire app
export const supabase = createClient(supabaseUrl, supabaseAnonKey);
