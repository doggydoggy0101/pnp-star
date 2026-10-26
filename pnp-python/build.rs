use std::path::{Path, PathBuf};

// Bake an rpath to the `mosek` crate's fetched libmosek64 so the cdylib resolves
// MOSEK at run time with no loader-path env var. (The mosek crate emits its own
// rpath, but `rustc-link-arg` from a dependency doesn't reach this artifact.)
fn mosek_rpath() {
    let Ok(out) = std::env::var("OUT_DIR") else {
        return;
    };
    let Some(build_dir) = Path::new(&out).ancestors().nth(2) else {
        return;
    };
    let mut stack: Vec<PathBuf> = std::fs::read_dir(build_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().starts_with("mosek-"))
                .unwrap_or(false)
        })
        .collect();
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let name = e.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("libmosek64")
                    && (name.ends_with(".dylib") || name.ends_with(".so"))
                {
                    if let Some(libdir) = p.parent() {
                        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir.display());
                        return;
                    }
                }
            }
        }
    }
}

fn main() {
    mosek_rpath();
}
