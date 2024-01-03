
#include "YAKL.h"

typedef double real;

using yakl::SArray;
using yakl::memDevice;
using yakl::memHost;
using yakl::Array;
using yakl::styleC;

YAKL_INLINE real constexpr operator"" _fp( long double x ) {
  return static_cast<real>(x);
}

typedef Array<real,1,memDevice,styleC> real1d;
typedef Array<real,2,memDevice,styleC> real2d;
typedef Array<real,3,memDevice,styleC> real3d;
typedef Array<real,4,memDevice,styleC> real4d;
typedef Array<real,5,memDevice,styleC> real5d;
typedef Array<real,6,memDevice,styleC> real6d;

int constexpr idR = 0;  // Density
int constexpr idU = 1;  // u-momentum
int constexpr idV = 2;  // v-momentum
int constexpr idW = 3;  // w-momentum
int constexpr idT = 4;  // Density * potential temperature
int constexpr ord = 5;
int constexpr hs = (ord-1)/2;

#include "TransformMatrices.h"
#include "WenoLimiter.h"

YAKL_INLINE static void modify_stencil_immersed_val( SArray<real,1,ord>       & stencil  ,
                                                     SArray<bool,1,ord> const & immersed ,
                                                     real                       val      ) {
  // Don't modify the stencils of immersed cells
  if (! immersed(hs)) {
    for (int ii=0; ii < ord; ii++) { if (immersed(ii)) stencil(ii) = val; }
  }
}

YAKL_INLINE static void modify_stencil_immersed_vect( SArray<real,1,ord>       & stencil  ,
                                                      SArray<bool,1,ord> const & immersed ,
                                                      SArray<real,1,ord> const & vect     ) {
  // Don't modify the stencils of immersed cells
  if (! immersed(hs)) {
    for (int ii=0; ii < ord; ii++) { if (immersed(ii)) stencil(ii) = vect(ii); }
  }
}

YAKL_INLINE static void modify_stencil_immersed_der0( SArray<real,1,ord>       & stencil  ,
                                                      SArray<bool,1,ord> const & immersed ) {
  // Don't modify the stencils of immersed cells
  if (! immersed(hs)) {
    for (int i2=hs+1; i2<ord; i2++) {
      if (immersed(i2)) { for (int i3=i2; i3<ord; i3++) { stencil(i3) = stencil(i2-1); }; break; }
    }
    for (int i2=hs-1; i2>=0 ; i2--) {
      if (immersed(i2)) { for (int i3=i2; i3>=0 ; i3--) { stencil(i3) = stencil(i2+1); }; break; }
    }
  }
}

template <class LIMITER, class PARAMS>
YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>     const & stencil      ,
                                                SArray<real,1,2  >           & gll          ,
                                                SArray<real,2,ord,2>   const & coefs_to_gll ,
                                                LIMITER                const & limiter      ,
                                                PARAMS                 const & weno_params  ) {
  SArray<real,1,ord> wenoCoefs;
  LIMITER::compute_limited_coefs( stencil , wenoCoefs , weno_params );
  for (int ii=0; ii<2; ii++) {
    real tmp = 0;
    for (int s=0; s < ord; s++) { tmp += coefs_to_gll(s,ii) * wenoCoefs(s); }
    gll(ii) = tmp;
  }
}

YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>   const & stencil     ,
                                                SArray<real,1,2  >         & gll         ,
                                                SArray<real,2,ord,2> const & sten_to_gll ) {
  for (int ii=0; ii<2; ii++) {
    real tmp = 0;
    for (int s=0; s < ord; s++) { tmp += sten_to_gll(s,ii) * stencil(s); }
    gll(ii) = tmp;
  }
}


int main() {
  yakl::init();
  {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    int constexpr num_state   = 5;
    int constexpr num_tracers = 2;
    int nz = 128;
    int ny = 32;
    int nx = 32;
    int nens = 1;
    real5d state   ("state"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);  state    = 1;
    real5d tracers ("tracers" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);  tracers  = 1;
    real4d pressure("pressure"            ,nz+2*hs,ny+2*hs,nx+2*hs,nens);  pressure = 1;
    SArray<real,2,ord,2> coefs_to_gll, sten_to_gll;
    TransformMatrices::coefs_to_gll_lower(coefs_to_gll);
    {
      SArray<real,2,ord,ord> s2c;
      TransformMatrices::sten_to_coefs(s2c);
      sten_to_gll = yakl::intrinsics::matmul_cr(coefs_to_gll,s2c);
    }
    real dx = 1;
    real dy = 1;
    real dz = 1;
    real r_dx = 1./dx;
    real r_dy = 1./dy;
    real r_dz = 1./dz;
  
    real6d state_limits_x   ("state_limits_x"   ,2,num_state  ,nz,ny,nx+1,nens);
    real6d tracers_limits_x ("tracers_limits_x" ,2,num_tracers,nz,ny,nx+1,nens);
    real5d pressure_limits_x("pressure_limits_x",2            ,nz,ny,nx+1,nens);
    real6d state_limits_y   ("state_limits_y"   ,2,num_state  ,nz,ny+1,nx,nens);
    real6d tracers_limits_y ("tracers_limits_y" ,2,num_tracers,nz,ny+1,nx,nens);
    real5d pressure_limits_y("pressure_limits_y",2            ,nz,ny+1,nx,nens);
    real6d state_limits_z   ("state_limits_z"   ,2,num_state  ,nz+1,ny,nx,nens);
    real6d tracers_limits_z ("tracers_limits_z" ,2,num_tracers,nz+1,ny,nx,nens);
    real5d pressure_limits_z("pressure_limits_z",2            ,nz+1,ny,nx,nens);

    limiter::WenoLimiter<ord> limiter(0.1,1,2,1,1.e3);
    auto weno_params = limiter.params;

    yakl::timer_start("main_loop");
    for (int iter = 0; iter < 1000; iter++) {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        {
          SArray<real,1,ord> stencil;
          SArray<real,1,2  > gll;
          // Density x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idR,hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_x(1,idR,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idR,k,j,i+1,iens) = gll(1);
          // Density y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idR,hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_y(1,idR,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idR,k,j+1,i,iens) = gll(1);
          // Density z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idR,k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_z(1,idR,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idR,k+1,j,i,iens) = gll(1);
        }
      #ifdef SPLIT_KERNELS
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
      #endif
        {
          SArray<real,1,ord> stencil;
          SArray<real,1,2  > gll;
          // u-vel x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idU,hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_x(1,idU,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idU,k,j,i+1,iens) = gll(1);
          // u-vel y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idU,hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_y(1,idU,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idU,k,j+1,i,iens) = gll(1);
          // u-vel z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idU,k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_z(1,idU,k  ,j,i,iens) = gll(0);
        }
      #ifdef SPLIT_KERNELS
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
      #endif
        {
          SArray<real,1,ord> stencil;
          SArray<real,1,2  > gll;
          // v-vel x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idV,hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_x(1,idV,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idV,k,j,i+1,iens) = gll(1);
          // v-vel y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idV,hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_y(1,idV,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idV,k,j+1,i,iens) = gll(1);
          // v-vel z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idV,k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_z(1,idV,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idV,k+1,j,i,iens) = gll(1);
        }
      #ifdef SPLIT_KERNELS
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
      #endif
        {
          SArray<real,1,ord> stencil;
          SArray<real,1,2  > gll;
          // w-vel x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idW,hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_x(1,idW,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idW,k,j,i+1,iens) = gll(1);
          // w-vel y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idW,hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_y(1,idW,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idW,k,j+1,i,iens) = gll(1);
          // w-vel z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idW,k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_z(1,idW,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idW,k+1,j,i,iens) = gll(1);
        }
      #ifdef SPLIT_KERNELS
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
      #endif
        {
          SArray<real,1,ord> stencil;
          SArray<real,1,2  > gll;
          // Theta x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idT,hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_x(1,idT,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idT,k,j,i+1,iens) = gll(1);
          // Theta y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idT,hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_y(1,idT,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idT,k,j+1,i,iens) = gll(1);
          // Theta z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idT,k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          state_limits_z(1,idT,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idT,k+1,j,i,iens) = gll(1);
        }
      #ifdef SPLIT_KERNELS
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
      #endif
        {
          SArray<real,1,ord> stencil;
          SArray<real,1,2  > gll;
          // Pressure x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = pressure(hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          pressure_limits_x(1,k,j,i  ,iens) = gll(0);
          pressure_limits_x(0,k,j,i+1,iens) = gll(1);
          // Pressure y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = pressure(hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          pressure_limits_y(1,k,j  ,i,iens) = gll(0);
          pressure_limits_y(0,k,j+1,i,iens) = gll(1);
          // Pressure z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = pressure(k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
          pressure_limits_z(1,k  ,j,i,iens) = gll(0);
          pressure_limits_z(0,k+1,j,i,iens) = gll(1);
        }
      #ifdef SPLIT_KERNELS
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
      #endif
        {
          #ifndef SPLIT_KERNELS
          for (int l=0; l < num_tracers; l++) {
          #endif
            SArray<real,1,ord> stencil;
            SArray<real,1,2  > gll;
            // Tracer x
            for (int ii=0; ii < ord; ii++) { stencil(ii) = tracers(l,hs+k,hs+j,i+ii,iens); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
            tracers_limits_x(1,l,k,j,i  ,iens) = gll(0);
            tracers_limits_x(0,l,k,j,i+1,iens) = gll(1);
            // Tracer y
            for (int jj=0; jj < ord; jj++) { stencil(jj) = tracers(l,hs+k,j+jj,hs+i,iens); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
            tracers_limits_y(1,l,k,j  ,i,iens) = gll(0);
            tracers_limits_y(0,l,k,j+1,i,iens) = gll(1);
            // Tracer z
            for (int kk=0; kk < ord; kk++) { stencil(kk) = tracers(l,k+kk,hs+j,hs+i,iens); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter,weno_params);
            tracers_limits_z(1,l,k  ,j,i,iens) = gll(0);
            tracers_limits_z(0,l,k+1,j,i,iens) = gll(1);
          #ifndef SPLIT_KERNELS
          }
          #endif
        }
      });

    }
    yakl::timer_stop("main_loop");
  }
  yakl::finalize();
}


