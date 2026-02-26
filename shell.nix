{
  pkgs ? import <nixpkgs> { },
}:
let
  inherit (pkgs) lib;
  python = (
    pkgs.python313.withPackages (ps: [
      ps.pre-commit-hooks
    ])
  );
in
pkgs.mkShell {
  # Specify Python version and dependencies
  packages = [
    python
    pkgs.basedpyright
    pkgs.nodejs
    pkgs.uv
    pkgs.ruff
    pkgs.glow
    pkgs.stdenv
    pkgs.pre-commit
  ];

  LD_LIBRARY_PATH = lib.makeLibraryPath (pkgs.pythonManylinuxPackages.manylinux1 ++ [ pkgs.zstd ]);
  UV_PYTHON_DOWNLOADS = "never";
  # Force uv to use nixpkgs Python interpreter
  UV_PYTHON_PREFERENCE = "only-system";
  UV_PYTHON = python;
  UV_SYSTEM_PYTHON = "true";

  shellHook = ''
    unset PYTHONPATH
    # Hack to use our torch rather than the uv one
    # export PYTHONPATH="${python}/${python.sitePackages}:$PYTHONPATH"
  '';
}
