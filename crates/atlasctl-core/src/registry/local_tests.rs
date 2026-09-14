// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::io::MemFileSystem;

#[test]
fn local_yaml_preserves_path_and_uses_filename_as_name() {
    for path in ["./recipes/custom.yaml", "/tmp/custom.yml", "custom.yaml"] {
        let fs = MemFileSystem::new();
        fs.insert(path, "model: org/m\ncontainer: img:tag\nruntime: atlas\n");
        let recipe = RegistrySet::builtin_only()
            .resolve_with_fs(&RecipeRef::parse(path), &fs)
            .expect("local recipe resolves");
        assert_eq!(recipe.name, "custom");
        assert_eq!(
            recipe.provenance,
            Provenance::LocalPath { path: path.into() }
        );
        recipe.launchable().expect("normal recipe validation");
    }
}

#[test]
fn local_errors_name_the_file_and_validate_content() {
    let fs = MemFileSystem::new();
    let set = RegistrySet::builtin_only();
    let reference = RecipeRef::parse("/tmp/custom.yaml");
    assert!(
        set.resolve_with_fs(&reference, &fs)
            .unwrap_err()
            .to_string()
            .contains("/tmp/custom.yaml")
    );
    for yaml in [
        "[",
        "container: img:tag\n",
        "model: --help\ncontainer: img:tag\n",
    ] {
        fs.insert("/tmp/custom.yaml", yaml);
        assert!(set.resolve_with_fs(&reference, &fs).is_err());
    }
    fs.insert(
        "/tmp/custom.yaml",
        "model: org/m\ncontainer: img:tag\nruntime: atlas\npre_exec: echo unsafe\n",
    );
    assert!(
        set.resolve_with_fs(&reference, &fs)
            .unwrap()
            .launchable()
            .is_err()
    );
}

#[test]
fn filesystem_resolution_preserves_builtin_resolution() {
    let fs = MemFileSystem::new();
    let recipe = RegistrySet::builtin_only()
        .resolve_with_fs(&RecipeRef::parse("qwen3.6-27b-fp8"), &fs)
        .expect("builtin resolves without filesystem data");
    assert!(matches!(recipe.provenance, Provenance::Builtin { .. }));
}

#[test]
fn local_recipe_renders_both_ep_ranks_with_pinned_weights() {
    use crate::chain::{Overrides, UserConfig};
    use crate::docker::collective::NcclRoce;
    use crate::docker::profile::{NvidiaDevices, ROOTLESS_V1};
    use crate::docker::translate::{LaunchContext, Placement, translate};
    use crate::host::{HostSnapshot, PosixUser};

    let fs = MemFileSystem::new();
    fs.insert("local.yaml", concat!(
        "model: nvidia/DeepSeek-V4-Flash-0731-NVFP4\n",
        "container: atlas-deepseek-v4-0731:bringup\nruntime: atlas\n",
        "min_nodes: 2\nmax_nodes: 2\nenv:\n  ATLAS_EP_GRAPHS: '0'\n",
        "defaults:\n  port: 8888\n  tensor_parallel: 1\n  ep_size: 2\n",
        "  max_model_len: 2048\n  max_batch_size: 1\n  kv_cache_dtype: fp8\n",
        "  gpu_memory_utilization: 0.90\n  oom_guard_mb: 512\n",
        "  swap_space_gb: 0\n  max_num_seqs: 1\n  max_prefill_tokens: 2048\n",
        "  speculative: false\n  enable_prefix_caching: false\n",
        "  model_from_path: /cache/huggingface/hub/models--nvidia--DeepSeek-V4-Flash-0731-NVFP4/snapshots/f1caa71142bd0be02f728c79f75042ac1e461579\n",
    ));
    let recipe = RegistrySet::builtin_only()
        .resolve_with_fs(&RecipeRef::parse("local.yaml"), &fs)
        .unwrap();
    let host = HostSnapshot {
        posix_user: Some(PosixUser {
            uid: 1000,
            gid: 1000,
        }),
        home: "/home/spark".into(),
        hf_cache_dir: "/home/spark/.cache/huggingface".into(),
        env: Default::default(),
    };
    let context = LaunchContext {
        profile: &ROOTLESS_V1,
        devices: &NvidiaDevices,
        collective: &NcclRoce,
    };
    for rank in [0, 1] {
        let plan = translate(
            &recipe,
            &Overrides::new(),
            &UserConfig::new(),
            &host,
            &Placement::Rank {
                rank,
                world_size: 2,
                master_addr: "10.10.10.1".into(),
                master_port: 29500,
            },
            &context,
        )
        .unwrap();
        assert!(plan.unmapped.is_empty());
        let command = plan.docker.to_string();
        assert!(command.contains(&format!("--rank {rank} --world-size 2")));
        assert!(command.contains(if rank == 0 { "--port 8888" } else { "--port 0" }));
        assert!(command.contains("--max-seq-len 2048"));
        assert!(command.contains("--swap-space-gb 0"));
        assert!(command.contains("--max-num-seqs 1"));
        assert!(command.contains("--max-prefill-tokens 2048"));
        assert!(command.contains("--model-from-path /cache/huggingface/hub/models--nvidia--DeepSeek-V4-Flash-0731-NVFP4/snapshots/f1caa71142bd0be02f728c79f75042ac1e461579"));
        assert!(!command.contains("--speculative"));
        assert!(!command.contains("--enable-prefix-caching"));
        assert_eq!(plan.docker.env["ATLAS_EP_GRAPHS"], "0");
    }
}
