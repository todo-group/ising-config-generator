#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <standards/timer.hpp>
#include <lattice/graph.hpp>
#include <cluster/union_find.hpp>

struct options {
  unsigned int seed, length;
  double beta_begin, beta_end, num_beta;
  unsigned int num_samples, therm, interval;
  std::string prefix;
  bool valid;

  options(unsigned int argc, char *argv[], bool print = true) :
    // default parameters
    seed(0), length(8), beta_begin(0.1), beta_end(1.0), num_beta(10),
    num_samples(10), therm(1 << 16), interval(1 << 10), prefix("ising"),
    valid(true) {
    for (unsigned int i = 1; i < argc; ++i) {
      switch (argv[i][0]) {
      case '-' :
        switch (argv[i][1]) {
        case 's' :
          if (++i == argc) { usage(print); return; }
          seed = std::atoi(argv[i]); break;
        case 'l' :
          if (++i == argc) { usage(print); return; }
          length = std::atoi(argv[i]); break;
        case 'b' :
          switch (argv[i][2]) {
          case 'b' :
            if (++i == argc) { usage(print); return; }
            beta_begin = std::atof(argv[i]); break;
          case 'e' :
            if (++i == argc) { usage(print); return; }
            beta_end = std::atof(argv[i]); break;
          case 'n' :
            if (++i == argc) { usage(print); return; }
            num_beta = std::atoi(argv[i]); break;
          default :
            usage(print); return;
          }
          break;
        case 'n' :
          if (++i == argc) { usage(print); return; }
          num_samples = std::atoi(argv[i]); break;
        case 'm' :
          switch (argv[i][2]) {
          case 't' :
            if (++i == argc) { usage(print); return; }
            therm = std::atoi(argv[i]); break;
          case 'i' :
            if (++i == argc) { usage(print); return; }
            interval = std::atoi(argv[i]); break;
          default :
            usage(print); return;
          }
          break;
        case 'p' :
          if (++i == argc) { usage(print); return; }
          prefix = std::string(argv[i]); break;
        case 'h' :
          usage(print, std::cout); return;
        default :
          usage(print); return;
        }
        break;
      default :
        usage(print); return;
      }
    }
    if (length == 0 || beta_begin <= 0 || beta_end <= 0 || num_beta == 0 || num_samples == 0 ||
        interval == 0) {
      std::cerr << "invalid parameter(s)\n"; usage(print); return;
    }
    if (seed == 0) {
      std::random_device rnd;
      seed = rnd();
    }
    if (print) {
      std::cout << "Seed of RNG (-s)             = " << seed << std::endl
                << "System linear size (-l)      = " << length << std::endl
                << "Beta (begin) (-bb)           = " << beta_begin << std::endl
                << "Beta (end) (-be)             = " << beta_end << std::endl
                << "Number of beta (-bn)         = " << num_beta << std::endl
                << "Number of samples (-n)       = " << num_samples << std::endl
                << "MCS for thermalization (-mt) = " << therm << std::endl
                << "MCS between samples (-mi)    = " << interval << std::endl
                << "Prefix (-p)                  = " << prefix << std::endl;
    }
  }
  void usage(bool print, std::ostream& os = std::cerr) {
    if (print)
      os << "[command line options]\n"
         << "  -s int     seed of RNG\n"
         << "  -l int     system linear size\n"
         << "  -bb double beta (begin)\n"
         << "  -be double beta (end)\n"
         << "  -bn int    number of beta\n"
         << "  -n int     number of samples\n"
         << "  -mt int    MCS for thermalization\n"
         << "  -mi int    MCS between samples\n"
         << "  -p string  prefix\n"
         << "  -h         this help\n";
    valid = false;
  }
};

int main(int argc, char* argv[]) {
  standards::timer tm;
  std::cout << "Swendsen-Wang Cluster Algorithm for Square Lattice Ising Model\n";
  options p(argc, argv);
  if (!p.valid) std::exit(127);
  
  // square lattice
  auto lattice = lattice::graph::simple(2, p.length);

  // random number generators
  std::mt19937 eng(p.seed);
  std::uniform_real_distribution<> r_uniform01;

  // spin configuration
  std::vector<int> spins(lattice.num_sites(), 1);

  // cluster information
  typedef cluster::union_find::node fragment_t;
  std::vector<fragment_t> fragments(lattice.num_sites());
  std::vector<int> flip(lattice.num_sites());

  // file
  std::ofstream ofslist(p.prefix + "-list.dat");
  std::ofstream ofsconf(p.prefix + "-conf.dat");
  
  for (unsigned int i = 0; i < p.num_beta; ++i) {
    double beta = p.beta_begin + i * (p.beta_end - p.beta_begin) / (p.num_beta - 1);
    std::cout << "beta = " << beta << std::endl;
    double prob = 1 - std::exp(-2 * beta);

    for (unsigned int j = 0; j < p.num_samples; ++j) {
      for (unsigned int mcs = 0; mcs < p.therm + p.interval * p.num_samples; ++mcs) {
        // initialize cluster information
        std::fill(fragments.begin(), fragments.end(), fragment_t());
        
        // cluster generation
        for (int b = 0; b < lattice.num_bonds(); ++b) {
          if (spins[lattice.source(b)] == spins[lattice.target(b)] && r_uniform01(eng) < prob)
            unify(fragments, lattice.source(b), lattice.target(b));
        }
        
        // assign cluster id & accumulate cluster properties
        int nc = 0;
        for (auto& f : fragments) {
          if (f.is_root()) f.set_id(nc++);
        }
        for (auto& f : fragments) f.set_id(cluster_id(fragments, f));

        // flip spins
        for (int c = 0; c < nc; ++c) flip[c] = (r_uniform01(eng) < 0.5);
        for (int s = 0; s < lattice.num_sites(); ++s)
          if (flip[fragments[s].id()]) spins[s] ^= 1;

        if (mcs >= p.therm && (mcs - p.therm) % p.interval == 0) {
          double ene = 0;
          for (int b = 0; b < lattice.num_bonds(); ++b) {
            ene -= (spins[lattice.source(b)] == spins[lattice.target(b)] ? 1.0 : -1.0);
          }
          double mag = 0;
          for (int s = 0; s < lattice.num_sites(); ++s) mag += 2 * spins[s] - 1;

          // output
          ofslist << p.length << '\t' << beta << '\t' << ene / lattice.num_sites() << '\t' << mag / lattice.num_sites() << std::endl;
          for (int s = 0; s < lattice.num_sites(); ++s)
            ofsconf << 2 * spins[s] - 1 << ' ';
          ofsconf << std::endl;
        }
      }
    }
  }

  double elapsed = tm.elapsed();
  std::cout << "Elapsed time = " << elapsed << " sec\n";
}
