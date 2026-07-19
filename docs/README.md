# Relay-OPD Project Page

Static project page for **Pass the Baton: Trajectory-Relayed On-Policy Distillation**, built on the [GithubPages-Template](https://github.com/ximinng/GithubPages-Template) by Ximing Xing (as used by [GUI-G2](https://zju-real.github.io/GUI-G2/)).

## Deploy

Copy the contents of this folder into the `zju-real/Relay-OPD` repository as the `docs/` directory, then enable GitHub Pages: *Settings → Pages → Deploy from a branch → `main` / `docs`*. The page will be served at <https://zju-real.github.io/Relay-OPD/>.

## Update figures

`assets/*.png` are exported from `iclr2026/figures/*.pdf` (ghostscript render at 220 dpi + white-margin crop). Re-export after figure changes and refresh the arXiv link in `index.html` once available (search for `id="arxiv"`).
