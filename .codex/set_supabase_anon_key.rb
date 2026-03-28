env_path = "/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence/.codex/worktrees/review-supabase-auth/.env"
anon_value = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlreHR1cWNmaHBhdWRkbmJ4cXlxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI3NzE5ODMsImV4cCI6MjA4ODM0Nzk4M30.6GNHRsRqjbj8jJ5iBLHNpLRS2uxgBZjoVv8twxPZdpg"

text = File.read(env_path)

if text.match?(/^SUPABASE_ANON_KEY=/)
  text.gsub!(/^SUPABASE_ANON_KEY=.*$/, "SUPABASE_ANON_KEY=#{anon_value}")
else
  text = "#{text}\nSUPABASE_ANON_KEY=#{anon_value}\n"
end

File.write(env_path, text)
