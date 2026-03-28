path = "/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence/.codex/worktrees/review-supabase-auth/web/static/index.html"
text = File.read(path)

pattern = /
\n?\s*<div class="instructions-section" id="analysis-identity-section" style="width:640px;margin-top:20px;display:none;">.*?
\s*<button type="button" class="btn-ghost" id="analysis-identity-signout">Sign out<\/button>\s*
\s*<\/div>\s*
\s*<\/div>
/mx

text.gsub!(pattern, "")

identity_block = <<~HTML.chomp
    <div class="instructions-section" id="analysis-identity-section" style="width:640px;margin-top:20px;display:none;">
      <div class="instructions-label">Analysis identity</div>
      <div id="analysis-identity-status" style="font-size:13px;color:var(--text-dim);margin-bottom:12px;">Loading identity…</div>
      <div id="analysis-identity-signed-out" style="display:none;">
        <input id="analysis-identity-email" class="instructions-input" type="email" maxlength="200" placeholder="name@rockaway.vc" autocomplete="email">
        <div style="display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap;">
          <button type="button" class="btn-ghost" id="analysis-identity-send">Send magic link</button>
          <div id="analysis-identity-pending" style="font-size:13px;color:var(--text-dim);display:none;"></div>
        </div>
      </div>
      <div id="analysis-identity-signed-in" style="display:none;">
        <div id="analysis-identity-user" style="font-size:14px;color:var(--text-light);margin-bottom:12px;"></div>
        <button type="button" class="btn-ghost" id="analysis-identity-signout">Sign out</button>
      </div>
    </div>
HTML

anchor = "    <div class=\"upload-zone\" id=\"upload-zone\">"
text.sub!(anchor, identity_block + "\n\n" + anchor)

File.write(path, text)
