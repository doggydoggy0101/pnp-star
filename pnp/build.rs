use std::path::{Path, PathBuf};

// Bake rpaths to every directory holding a `libmosek64`, so a binary resolves
// MOSEK at run time with no loader-path env var. The `mosek` crate emits its own
// rpath, but `rustc-link-arg` from a dependency does not reach downstream
// binaries, so each runnable crate must re-emit it. For crates depending on
// `pnp`, the directories are also exported as `DEP_PNP_MOSEK_RPATH` (via
// `links = "pnp"`), which their own build.rs turns into rpaths.
//
// ! The download must exist before this runs. Cargo orders build scripts only
// across `links` packages and build-dependencies, and `mosek` has no `links`,
// so pnp/Cargo.toml also lists it under [build-dependencies].
//
// ! Match on the `libmosek64` prefix only: the version sits in a different
// place in the file name on each platform. Emitting every candidate is fine —
// the loader matches on soname, so directories holding another version are
// simply skipped.

/// `<base>/<version>/tools/platform/<platform>/bin` for every version found.
fn bin_dirs(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(versions) = std::fs::read_dir(base) else {
        return out;
    };
    for v in versions.flatten() {
        let platforms = v.path().join("tools").join("platform");
        let Ok(rd) = std::fs::read_dir(&platforms) else {
            continue;
        };
        for p in rd.flatten() {
            let bin = p.path().join("bin");
            if bin.is_dir() {
                out.push(bin);
            }
        }
    }
    out
}

pub fn mosek_rpath() {
    let mut roots: Vec<PathBuf> = Vec::new();

    // a distro the mosek crate downloaded: <target>/<profile>/build/mosek-*/out/mosek
    if let Ok(out) = std::env::var("OUT_DIR") {
        // OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out
        if let Some(build_dir) = Path::new(&out).ancestors().nth(2) {
            if let Ok(rd) = std::fs::read_dir(build_dir) {
                for e in rd.flatten() {
                    let name = e.file_name();
                    if name.to_string_lossy().starts_with("mosek-") {
                        roots.push(e.path().join("out").join("mosek"));
                    }
                }
            }
        }
    }
    // an installation the mosek crate found instead of downloading
    for var in ["MOSEK_INST_BASE", "HOME"] {
        if let Some(v) = std::env::var_os(var) {
            roots.push(PathBuf::from(v).join("mosek"));
        }
    }

    let mut dirs: Vec<PathBuf> = Vec::new();
    for root in &roots {
        for d in bin_dirs(root) {
            if !dirs.contains(&d) {
                dirs.push(d);
            }
        }
    }
    // MOSEK_BINDIR_<major><minor>, which the mosek crate checks first
    for (k, v) in std::env::vars_os() {
        if k.to_string_lossy().starts_with("MOSEK_BINDIR_") {
            let d = PathBuf::from(v);
            if d.is_dir() && !dirs.contains(&d) {
                dirs.push(d);
            }
        }
    }

    for d in &dirs {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }
    // DEP_PNP_MOSEK_RPATH for dependents, joined like PATH
    if !dirs.is_empty() {
        if let Ok(joined) = std::env::join_paths(&dirs) {
            println!("cargo:mosek_rpath={}", joined.to_string_lossy());
        }
    }
}

fn main() {
    mosek_rpath();
}
