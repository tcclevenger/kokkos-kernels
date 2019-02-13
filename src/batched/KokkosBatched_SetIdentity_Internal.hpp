#ifndef __KOKKOSBATCHED_SET_IDENTITY_INTERNAL_HPP__
#define __KOKKOSBATCHED_SET_IDENTITY_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"


namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Serial Internal Impl
    /// ==================== 
    struct SerialSetIdentityInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const int m, 
             /* */ ValueType *__restrict__ A, const int as0, const int as1) {
        const ValueType one(1), zero(0);
        for (int j=0;j<m;++j) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
          for (int i=0;i<m;++i) {
            A[i*as0+j*as1] = i == j ? one : zero;
          }
        }
        
        return 0;
      }
    };

    ///
    /// Team Internal Impl
    /// ==================
    template<typename MemberType>
    struct TeamSetIdentityInternal {
      template<typename ValueType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(MemberType const int m, 
             /* */ ValueType *__restrict__ A, const int as0, const int as1) {
        const ValueType one(1), zero(0);
        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member,0,m),
           [&](const int &i) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int j=0;j<m;++j) 
          A[i*as0+j*as1] = i == j ? one : zero;
          });
        
        return 0;
      }
    };

  }//  end namespace Experimental
} // end namespace KokkosBatched


#endif
