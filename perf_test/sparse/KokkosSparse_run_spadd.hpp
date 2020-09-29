/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosKernels_TestParameters.hpp"

namespace KokkosKernels {
namespace Experiment {

template <typename crsMat_t>
bool validate_spadd(const crsMat_t& A, const crsMat_t& B, const crsMat_t& C)
{
  //typedef typename crsMat_t::row_map_type::non_const_type row_map_type;
  //typedef typename crsMat_t::index_type::non_const_type entries_type;
  //typedef typename crsMat_t::values_type::non_const_type values_type;
  typedef typename crsMat_t::ordinal_type lno_t;
  typedef typename crsMat_t::size_type size_type;
  typedef typename crsMat_t::value_type scalar_t;
  //typedef typename crsMat_t::device_type device_t;
  //typedef typename device_t::execution_space exec_space;
  //typedef typename device_t::memory_space mem_space;

  lno_t numRows = A.numRows();
  lno_t numCols = A.numCols();

  //typedef KokkosKernels::Experimental::KokkosKernelsHandle<size_type, lno_t, scalar_t, exec_space, mem_space, mem_space> KernelHandle;

  auto Avalues = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.values);
  auto Arowmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.row_map);
  auto Aentries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.entries);
  auto Bvalues = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B.values);
  auto Browmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B.graph.row_map);
  auto Bentries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B.graph.entries);
  auto Cvalues = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C.values);
  auto Crowmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C.graph.row_map);
  auto Centries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C.graph.entries);
  using KAT = Kokkos::ArithTraits<scalar_t>;
  auto zero = KAT::zero();
  auto eps = KAT::epsilon();
  //check that C is correct and sorted, row-by-row
  for(lno_t row = 0; row < numRows; row++)
  {
    std::vector<scalar_t> correct(numCols, zero);
    std::vector<bool> nonzeros(numCols, false);
    for(size_type i = Arowmap(row); i < Arowmap(row + 1); i++)
    {
      correct[Aentries(i)] += Avalues(i);
      nonzeros[Aentries(i)] = true;
    }
    for(size_type i = Browmap(row); i < Browmap(row + 1); i++)
    {
      correct[Bentries(i)] += Bvalues(i);
      nonzeros[Bentries(i)] = true;
    }
    size_type nz = 0;
    for(lno_t i = 0; i < numCols; i++)
    {
      if(nonzeros[i])
        nz++;
    }
    //make sure C has the right number of entries
    auto actualNZ = Crowmap(row + 1) - Crowmap(row);
    if(actualNZ != nz)
    {
      std::cout << "A+B row " << row << " has " << actualNZ << " entries but should have " << nz << '\n';
      return false;
    }
    //make sure C's indices are sorted
    for(size_type i = Crowmap(row) + 1; i < Crowmap(row + 1); i++)
    {
      if(Centries(i - 1) > Centries(i))
      {
        std::cout << "C row " << row << " is not sorted\n";
        return false;
      }
    }
    //make sure C has the correct values
    for(size_type i = Crowmap(row); i < Crowmap(row + 1); i++)
    {
      scalar_t Cval = Cvalues(i);
      lno_t Ccol = Centries(i);
      //Check that result is correct to 1 ULP
      if(KAT::abs(correct[Ccol] - Cval) > KAT::abs(correct[Ccol] * eps))
      {
        std::cout << "A+B row " << row << ", column " << Ccol << " has value " << Cval << " but should be " << correct[Ccol] << '\n';
        return false;
      }
    }
  }
  return true;
}

template <typename crsMat_t>
void run_experiment(Parameters params)
{
  using namespace KokkosSparse;
  using namespace KokkosSparse::Experimental;

  using size_type = typename crsMat_t::size_type;
  using lno_t = typename crsMat_t::ordinal_type;
  using scalar_t = typename crsMat_t::value_type;
  using device_t = typename crsMat_t::device_type;
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;

  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle
      <size_type, lno_t, scalar_t, exec_space, mem_space, mem_space>;

  std::cout << "************************************* \n";
  std::cout << "************************************* \n";
  std::cout << "Loading A from " << params.a_mtx_bin_file << '\n';
  crsMat_t A = Impl::read_kokkos_crst_matrix<crsMat_t>(params.a_mtx_bin_file);
  std::cout << "Loading B from " << params.b_mtx_bin_file << '\n';
  crsMat_t B = Impl::read_kokkos_crst_matrix<crsMat_t>(params.b_mtx_bin_file);
  //Make sure dimensions are compatible
  if(A.numRows() != B.numRows())
  {
    std::cout << "ERROR: A and B have different numbers of rows\n";
    exit(1);
  }
  if(A.numCols() != B.numCols())
  {
    std::cout << "ERROR: A and B have different numbers of columns\n";
    exit(1);
  }
  lno_t m = A.numRows();
  lno_t n = A.numCols();
  std::cout << "Read in A and B: " << m << "x" << n << '\n';

  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
  typedef typename crsMat_t::StaticCrsGraphType::row_map_type::non_const_type lno_view_t;
  typedef typename crsMat_t::StaticCrsGraphType::entries_type::non_const_type lno_nnz_view_t;
  typedef typename crsMat_t::StaticCrsGraphType::row_map_type const_lno_view_t;
  typedef typename crsMat_t::StaticCrsGraphType::entries_type const_lno_nnz_view_t;

  lno_view_t row_mapC;
  lno_nnz_view_t entriesC;
  scalar_view_t valuesC;

  std::cout << "Computing sum with types:\n";
  std::cout << "Exec: " << typeid(exec_space).name() << '\n';
  std::cout << "Mem: " << typeid(mem_space).name() << '\n';
  std::cout << "lno_t: " << typeid(lno_t).name() << '\n';
  std::cout << "size_type: " << typeid(size_type).name() << '\n';

  if(params.assume_sorted)
    std::cout << "Assuming input matrices are sorted.\n";
  else
    std::cout << "Assuming input matrices are not sorted.\n";
  double symbolic_time = 0;
  double numeric_time = 0;

  size_type c_nnz = 0;

  for(int rep = 0; rep < params.repeat; rep++)
  {
    KernelHandle kh;

    kh.create_spadd_handle(params.assume_sorted);
    auto addHandle = kh.get_spadd_handle();

    row_mapC = lno_view_t("non_const_lnow_row", m + 1);

    Kokkos::Impl::Timer timer1;

    spadd_symbolic<KernelHandle, const_lno_view_t, const_lno_nnz_view_t, const_lno_view_t, const_lno_nnz_view_t, lno_view_t, lno_nnz_view_t>
      (&kh,
        A.graph.row_map, A.graph.entries,
        B.graph.row_map, B.graph.entries,
        row_mapC);

    exec_space().fence();
    symbolic_time += timer1.seconds();

    c_nnz = addHandle->get_max_result_nnz();

    entriesC = lno_nnz_view_t(Kokkos::ViewAllocateWithoutInitializing("entriesC (empty)"), c_nnz);
    valuesC = scalar_view_t("valuesC (empty)", c_nnz);

    Kokkos::Impl::Timer timer3;

    spadd_numeric(&kh,
        A.graph.row_map, A.graph.entries, A.values, 1.0, //A, alpha
        B.graph.row_map, B.graph.entries, B.values, 1.0, //B, beta
        row_mapC, entriesC, valuesC);  //C

    exec_space().fence();
    numeric_time += timer3.seconds();

    if(rep % 100 == 99)
      std::cout << "Finished " << rep + 1 << " spadd repetitions\n";
  }

  std::cout
    << "total_time:" << (symbolic_time + numeric_time) / params.repeat
    << " symbolic_time:" << symbolic_time / params.repeat
    << " numeric_time:" << numeric_time / params.repeat << std::endl;

  if (params.verbose)
  {
    std::cout << "row_mapC:" << row_mapC.extent(0) << std::endl;
    std::cout << "entriesC:" << entriesC.extent(0) << std::endl;
    std::cout << "valuesC:" << valuesC.extent(0) << std::endl;
    KokkosKernels::Impl::print_1Dview(valuesC);
    KokkosKernels::Impl::print_1Dview(entriesC);
    KokkosKernels::Impl::print_1Dview(row_mapC);
    std::cout << "Validating result...";
    crsMat_t C("C", m, n, c_nnz, valuesC, row_mapC, entriesC);
    if(validate_spadd(A, B, C))
    {
      std::cout << "OK.\n";
    }
    else
    {
      std::cout << '\n';
      std::cout << "**** RESULT INCORRECT!\n";
    }
  }
  if(params.c_mtx_bin_file)
  {
    std::cout << "Writing C (" << m << "x" << n << ") to " << params.c_mtx_bin_file << "\n";
    crsMat_t C("C", m, n, c_nnz, valuesC, row_mapC, entriesC);
    Impl::write_kokkos_crst_matrix<crsMat_t>(C, params.c_mtx_bin_file);
  }
}

}}  // namespace KokkosKernels::Experiment
