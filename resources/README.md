# Language resources

Morphological decomposition reads external UniMorph files. These files are not bundled with the code. Set `UNIMORPH_ROOT` to a directory containing language subdirectories such as `eng/eng`, `spa/spa`, `ara/ara`, `deu/deu`, and `rus/rus`. The default location is `resources/unimorph/` in this repository.

```bash
export UNIMORPH_ROOT="/absolute/path/to/unimorph"
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet
```

English decomposition also requires a system Enchant library with an English dictionary. For example, install `enchant` with your system package manager, then confirm that `python -c 'import enchant; print(enchant.list_languages())'` lists an English locale.

UniMorph is distributed through the [UniMorph project](https://unimorph.github.io/). Preserve the license and revision of every language resource you download. Derivation files are only needed for workflows that explicitly include derivations; the final post-hoc language-modeling setup excludes them.

Model checkpoints and datasets are downloaded separately from their owners. Some models require an approved Hugging Face account. Authenticate with Hugging Face's login tool or an environment token; do not put tokens into command files tracked by Git.

