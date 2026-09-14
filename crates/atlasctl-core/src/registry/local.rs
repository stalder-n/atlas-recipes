// SPDX-License-Identifier: AGPL-3.0-only

//! Local recipe resolution through the caller's filesystem boundary.

use super::{RecipeRef, RegistrySet};
use crate::io::FileSystem;
use crate::recipe::{Provenance, Recipe};
use anyhow::{Context, Result};
use std::path::Path;

impl RegistrySet {
    /// Resolve registry references normally, or read a local YAML recipe.
    pub fn resolve_with_fs(&self, reference: &RecipeRef, fs: &dyn FileSystem) -> Result<Recipe> {
        let RecipeRef::Path(path) = reference else {
            return self.resolve(reference).map_err(Into::into);
        };
        let file = Path::new(path);
        let name = file
            .file_stem()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .with_context(|| format!("recipe path has no filename: {path}"))?;
        let yaml = fs.read_to_string(file)?;
        Recipe::parse(name, &yaml, Provenance::LocalPath { path: path.clone() })
            .with_context(|| format!("could not load local recipe {path}"))
    }
}
