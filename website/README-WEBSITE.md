# Voxa website

Static website for **voxaai.me**, hosted from `crhy/voxa` with Cloudflare Pages.

The source files are in `website/`. `build.sh` copies the site and the repository's current logo and assistant screenshot into `website/dist/`. This keeps the deployed output limited to public website assets.

## Cloudflare Pages settings

- Connect GitHub repository: `crhy/voxa`
- Production branch: `main` (after this pull request is merged)
- Framework preset: **None**
- Root directory: repository root (leave blank)
- Build command: `bash website/build.sh`
- Build output directory: `website/dist`
- Custom domain: `voxaai.me`

Test a local build with `bash website/build.sh`, then inspect `website/dist/index.html`. The generated directory is ignored by Git.

Replace the screenshot reference in `index.html` and the corresponding copy in `build.sh` when new media is ready.
