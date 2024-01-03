# To Reproduce the issue

## With split kernels
```bash
git clone git@github.com:mrnorman/recon_kernel.git
cd recon_kernel/build
git submodule update --init
source crusher_gpu.env
./cmakescript.sh
make -j
./recon_kernel
```

## Without split kernels
Edit `crusher_gpu.env` and remove `-DSPLIT_KERNELS` from the `YAKL_HIP_FLAGS` environment variables, and repeat the previous workflow

