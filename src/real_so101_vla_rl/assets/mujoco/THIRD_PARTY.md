# Third-party MuJoCo assets

The `SO101/` and `SO101_menagerie/` directories are byte-for-byte copies from
`so101-nexus` commit `85d7ee986c4af0d769259c89ed97ae409db9c2cb`.

- Source project: `so101-nexus`
- Source paths: `src/so101_nexus/assets/SO101` and
  `src/so101_nexus/assets/SO101_menagerie`
- Project license: Apache License 2.0; see `SO101_NEXUS_LICENSE.md`.
- Menagerie model provenance and its pinned upstream revision are retained in
  `SO101_menagerie/PROVENANCE.md`.
- Per-file checksums: `SO101_NEXUS_SHA256SUMS`.

Do not modify files inside either vendored directory. Competition-specific
coordinate adaptation belongs in `competition_2026/so101_competition.xml`.
