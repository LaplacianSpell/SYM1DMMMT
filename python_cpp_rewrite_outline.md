# SYM1DMMMT: Python and C/C++ Rewrite Outline

## Project Overview

This repository implements the **BFSS Matrix Model** (Banks-Fischler-Shenker-Susskind) simulation on a lattice, originally written in Fortran. The code performs Monte Carlo simulations of supersymmetric matrix quantum mechanics with both bosonic and fermionic degrees of freedom.

### Current Architecture Analysis

**Core Components:**
- **Bosonic matrices**: `xmat(nmat×nmat×ndim×nsite)` - 9 Hermitian matrices
- **Fermionic fields**: `pf(nmat×nmat×nspin×nsite×npf)` - Pseudo-fermion fields 
- **Gauge field**: `alpha(nmat)` - U(1) gauge degrees of freedom
- **Lattice**: 1D temporal lattice with `nsite=24` points
- **Matrix size**: `nmat=8` (SU(N) matrices)
- **Spinor structure**: `nspin=16` (10D spinors)

**Key Algorithms:**
- Hybrid Monte Carlo (HMC) evolution
- Rational Hybrid Monte Carlo (RHMC) for fermions
- Multi-mass BiCG solver for linear systems
- Fourier acceleration techniques
- MPI parallelization for matrix operations
- GPU acceleration with OpenACC

## Proposed Python and C/C++ Architecture

### 1. Core Language Distribution

**Python Components (High-level orchestration):**
- Main simulation driver
- Configuration management
- I/O operations and data serialization
- Plotting and analysis tools
- Parameter sweeps and job management
- Unit testing framework

**C/C++ Components (Performance-critical kernels):**
- Matrix operations and linear algebra
- HMC/RHMC evolution algorithms
- BiCG solver implementations
- Fourier transforms
- Dirac operator applications
- CUDA/OpenCL kernels for GPU acceleration

### 2. Detailed Module Structure

#### 2.1 Python Module Structure

```
bfss_simulation/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── simulation.py           # Main simulation orchestrator
│   ├── configuration.py        # Parameter management
│   ├── lattice.py              # Lattice geometry and boundary conditions
│   └── measurements.py         # Observable calculations
├── io/
│   ├── __init__.py
│   ├── config_io.py           # Configuration file handling
│   ├── data_output.py         # Results serialization (HDF5/NetCDF)
│   └── checkpoint.py          # Simulation state management
├── analysis/
│   ├── __init__.py
│   ├── observables.py         # Physics observables
│   ├── statistics.py          # Statistical analysis
│   └── plotting.py            # Visualization tools
├── algorithms/
│   ├── __init__.py
│   ├── hmc.py                 # HMC interface wrapper
│   ├── rhmc.py                # RHMC interface wrapper
│   └── solvers.py             # Linear solver interfaces
└── utils/
    ├── __init__.py
    ├── random.py              # Random number generation
    ├── fourier.py             # FFT utilities
    └── mpi_utils.py           # MPI coordination
```

#### 2.2 C/C++ Library Structure

```
libbfss/
├── include/
│   ├── bfss_types.h           # Data structure definitions
│   ├── matrix_ops.h           # Matrix operation interfaces
│   ├── hmc_core.h             # HMC evolution kernels
│   ├── dirac_operator.h       # Fermion operator definitions
│   ├── solvers.h              # Linear algebra solvers
│   └── gpu_kernels.h          # CUDA/OpenCL interfaces
├── src/
│   ├── matrix_operations.cpp   # Dense matrix operations
│   ├── hmc_evolution.cpp       # Molecular dynamics
│   ├── rhmc_evolution.cpp      # Rational HMC for fermions
│   ├── dirac_operator.cpp      # Fermion operator implementations
│   ├── bicg_solver.cpp         # Multi-mass BiCG solver
│   ├── fourier_accel.cpp       # Fourier acceleration
│   └── measurements_core.cpp   # Core observable calculations
├── gpu/
│   ├── cuda_kernels.cu         # CUDA implementations
│   ├── matrix_ops_gpu.cu       # GPU matrix operations
│   └── solver_gpu.cu           # GPU-accelerated solvers
└── tests/
    ├── test_matrix_ops.cpp
    ├── test_solvers.cpp
    └── benchmark_kernels.cpp
```

### 3. Key Data Structures

#### 3.1 C/C++ Core Types

```cpp
// Core configuration structure
struct BFSSConfig {
    int nsite;              // Lattice size
    int nmat;               // Matrix dimension
    int ndim;               // Spatial dimensions (9)
    int nspin;              // Spinor components (16)
    int npf;                // Number of pseudofermion fields
    int nremez_md, nremez_pf; // Remez coefficients
    
    // Physical parameters
    double temperature;
    double flux;
    double g_alpha;         // Alpha constraint coupling
    double g_R, RCUT;       // R^2 constraint parameters
    
    // Evolution parameters
    int ntau, nratio;
    double dtau_xmat, dtau_alpha;
    
    // Solver parameters
    int max_iteration;
    double max_err;
};

// Matrix field structure
struct MatrixField {
    std::complex<double>* data;
    int nmat, ndim, nsite;
    int margin;  // Ghost sites for boundary conditions
    
    // Memory layout: [imat][jmat][idim][isite+margin]
    std::complex<double>& operator()(int i, int j, int d, int t);
    void set_boundary_conditions(int bc_type);
};

// Fermion field structure  
struct FermionField {
    std::complex<double>* data;
    int nmat, nspin, nsite, npf;
    int margin;
    
    // Memory layout: [imat][jmat][ispin][isite+margin][ipf]
    std::complex<double>& operator()(int i, int j, int s, int t, int pf);
};
```

#### 3.2 Python Interface Classes

```python
class BFSSSimulation:
    """Main simulation driver"""
    def __init__(self, config_file: str):
        self.config = Configuration(config_file)
        self.lattice = Lattice(self.config)
        self.fields = FieldConfiguration(self.config)
        self._core = libbfss.BFSSCore(self.config.to_dict())
    
    def run_trajectory(self) -> dict:
        """Execute one HMC trajectory"""
        
    def run_simulation(self, n_trajectories: int) -> None:
        """Run full simulation"""
        
    def measure_observables(self) -> dict:
        """Calculate physical observables"""

class FieldConfiguration:
    """Container for all field variables"""
    def __init__(self, config: Configuration):
        self.xmat = MatrixField(config.nmat, config.ndim, config.nsite)
        self.alpha = np.zeros(config.nmat, dtype=np.float64)
        self.pf = [FermionField(config) for _ in range(config.npf)]
    
    def save(self, filename: str) -> None:
        """Save configuration to file"""
        
    def load(self, filename: str) -> None:
        """Load configuration from file"""
```

### 4. Implementation Strategy

#### 4.1 Phase 1: Core Infrastructure (Weeks 1-3)

**C/C++ Foundation:**
- Set up CMake build system with dependencies (BLAS/LAPACK, FFTW, MPI)
- Implement basic data structures and memory management
- Create Python bindings using pybind11
- Port basic matrix operations and tests

**Python Framework:**
- Create package structure and configuration system
- Implement I/O utilities for HDF5/NetCDF formats
- Set up pytest testing framework
- Create basic visualization tools

#### 4.2 Phase 2: Core Algorithms (Weeks 4-8)

**Linear Algebra Core:**
```cpp
// High-performance matrix operations
class MatrixOperations {
public:
    static void multiply_hermitian(const MatrixField& A, const MatrixField& B, 
                                   MatrixField& C);
    static void commutator(const MatrixField& A, const MatrixField& B, 
                           MatrixField& result);
    static double trace_squared(const MatrixField& field);
    static void hermitian_projection(MatrixField& field);
};

// Dirac operator for fermions
class DiracOperator {
public:
    DiracOperator(const BFSSConfig& config);
    void apply(const FermionField& input, FermionField& output);
    void apply_dagger(const FermionField& input, FermionField& output);
    void set_gauge_field(const MatrixField& xmat, const double* alpha);
};
```

**HMC Evolution:**
```cpp
class HMCEvolution {
public:
    HMCEvolution(const BFSSConfig& config);
    
    // Molecular dynamics integration
    void leapfrog_step(MatrixField& xmat, double* alpha,
                       MatrixField& p_xmat, double* p_alpha,
                       double dtau);
    
    // Force calculations
    void calculate_force_bosonic(const MatrixField& xmat, const double* alpha,
                                 MatrixField& force_xmat, double* force_alpha);
    void calculate_force_fermionic(const MatrixField& xmat, const double* alpha,
                                   const FermionField& pf,
                                   MatrixField& force_xmat, double* force_alpha);
    
    double calculate_hamiltonian(const MatrixField& xmat, const double* alpha,
                                 const MatrixField& p_xmat, const double* p_alpha);
};
```

#### 4.3 Phase 3: Advanced Features (Weeks 9-12)

**Multi-mass BiCG Solver:**
```cpp
class BiCGSolver {
public:
    struct SolverParams {
        int max_iterations;
        double tolerance;
        std::vector<double> masses;  // Remez coefficients
    };
    
    int solve_multimass(const DiracOperator& dirac_op,
                        const FermionField& source,
                        std::vector<FermionField>& solutions,
                        const SolverParams& params);
};
```

**GPU Acceleration:**
```cpp
#ifdef USE_CUDA
class GPUKernels {
public:
    static void matrix_multiply_gpu(const MatrixField& A, const MatrixField& B,
                                    MatrixField& C, cudaStream_t stream);
    static void dirac_operator_gpu(const FermionField& input, FermionField& output,
                                   const MatrixField& gauge, cudaStream_t stream);
};
#endif
```

#### 4.4 Phase 4: Parallel Computing (Weeks 13-16)

**MPI Parallelization:**
```python
class MPISimulation(BFSSSimulation):
    """MPI-parallel version of the simulation"""
    def __init__(self, config_file: str, comm: MPI.Comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        super().__init__(config_file)
        self._setup_domain_decomposition()
    
    def _setup_domain_decomposition(self):
        """Distribute matrix indices across MPI ranks"""
        
    def synchronize_fields(self):
        """Communicate boundary data between processes"""
```

### 5. Performance Optimization Strategy

#### 5.1 Memory Layout Optimization
- Use column-major ordering for BLAS compatibility
- Implement memory pools for temporary arrays
- SIMD vectorization for inner loops
- Cache-friendly data access patterns

#### 5.2 Computational Kernels
```cpp
// Optimized matrix multiplication kernel
void matrix_multiply_optimized(const complex_matrix& A, const complex_matrix& B,
                               complex_matrix& C) {
    // Use OpenMP for parallelization
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nmat; ++i) {
        for (int j = 0; j < nmat; ++j) {
            // SIMD-optimized inner product
            std::complex<double> sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nmat; ++k) {
                sum += A(i,k) * B(k,j);
            }
            C(i,j) = sum;
        }
    }
}
```

### 6. Testing and Validation Strategy

#### 6.1 Unit Tests
```python
def test_matrix_operations():
    """Test basic matrix operations against reference"""
    config = create_test_config()
    field_a = create_random_matrix_field(config)
    field_b = create_random_matrix_field(config)
    
    # Test C++ implementation
    result_cpp = libbfss.matrix_multiply(field_a, field_b)
    
    # Test against NumPy reference
    result_numpy = numpy_matrix_multiply(field_a, field_b)
    
    assert np.allclose(result_cpp, result_numpy, rtol=1e-12)

def test_solver_convergence():
    """Test BiCG solver convergence"""
    # Create test system with known solution
    # Verify solver finds correct solution within tolerance
```

#### 6.2 Integration Tests
- Compare observables with original Fortran code
- Benchmark performance against reference implementation
- Test parallel scaling efficiency
- Validate GPU acceleration correctness

### 7. Build System and Dependencies

#### 7.1 CMake Configuration
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(libbfss VERSION 1.0.0 LANGUAGES CXX)

# Dependencies
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)

# Optional GPU support
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(USE_CUDA ON)
endif()

# Python bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(libbfss ${SOURCES})
```

#### 7.2 Python Package Setup
```python
# setup.py
from setuptools import setup, find_packages
from cmake_setuptools import CMakeExtension, CMakeBuild

setup(
    name="bfss-simulation",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=[CMakeExtension("libbfss")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
        "h5py>=3.0.0",
        "matplotlib>=3.3.0",
        "mpi4py>=3.0.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
        "analysis": ["pandas", "seaborn", "jupyter"],
    }
)
```

### 8. Migration Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Phase 1** | 3 weeks | Core infrastructure, basic Python framework |
| **Phase 2** | 4 weeks | Matrix operations, HMC evolution, basic solver |
| **Phase 3** | 4 weeks | Advanced solvers, GPU kernels, optimization |
| **Phase 4** | 4 weeks | MPI parallelization, full feature parity |
| **Phase 5** | 1 week | Documentation, packaging, release |

### 9. Expected Benefits

**Performance Improvements:**
- 2-5x speedup from optimized C++ kernels
- 10-50x speedup with GPU acceleration
- Better scaling with modern MPI implementations

**Maintainability:**
- Modern C++17/20 features for better code organization
- Python high-level interface for easier experimentation
- Comprehensive testing and documentation
- Modular design for easier extension

**Usability:**
- Interactive Jupyter notebook support
- Modern visualization and analysis tools
- Easier parameter sweeps and job management
- Better integration with ML/AI workflows

This architecture provides a solid foundation for a high-performance, maintainable implementation of the BFSS matrix model simulation while preserving all the physics and numerical algorithms of the original Fortran code.