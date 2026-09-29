#pragma once

#include "HostShimCAPI.h"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <random>
#include <string>
#include <array>

#define MAX_RESULT_VALUES_TO_PRINT 32
template <int N>
struct MemRefDescriptor {
  void *base;
  void *data;
  int64_t offset = 0;
  int64_t shape[N];
  int64_t strides[N];
} __attribute__((aligned(128)));

template <int Rank, int NumInputs, int NumOutputs>
class TestModel {
public:
  TestModel(std::string name, int version, bool verbose = false)
      : name_(name), version_(version), verbose_(verbose) {}

  ~TestModel() {
    // free memory
    for (int i = 0; i < NumInputs; i++)
      tsi_dealloc(inputs[i].base);
    for (int i = 0; i < NumOutputs; i++)
      tsi_dealloc(outputs[i].base);
    tsi_finalize();
  }

  template <typename ElType>
  void initRandom(size_t numElements,
                  std::array<int, 2> inputRange = {-10, 10}) {
    static_assert(Rank == 1,
                  "initRandom(size_t) is only defined for Rank == 1");
    size_t inputSizes[2][Rank] = {{numElements}, {numElements}};
    size_t outputSizes[1][Rank] = {{numElements}};
    init<ElType, ElType>(inputSizes, outputSizes,
                         /*initWithRandom=*/true, inputRange);
  }

#if 0
  template <typename ElType>
  void initFill(size_t numElements, ElType val) {
    static_assert(Rank == 1,
                  "initRandom(size_t) is only defined for Rank == 1");
    size_t inputSizes[2][Rank] = {{numElements}, {numElements}};
    size_t outputSizes[1][Rank] = {{numElements}};
    init<ElType, ElType>(inputSizes, outputSizes);
    for (int i = 0; i < NumInputs; i++) {
      auto nEls = getNumElements(inputs[i]);
      for (size_t j = 0; j < nEls; j++)
        static_cast<ElType *>(inputs[i].data)[j] = val;
    }
  }
#endif /* 0 */

  template <typename InputsElType, typename OutputsElType>
  void init(size_t inputSizes[NumInputs][Rank],
            size_t outputSizes[NumOutputs][Rank], bool initWithRandom = false,
            std::array<int, 2> inputRange = {-10, 10}) {
    tsi_initialize(1);

    for (int i = 0; i < NumInputs; i++)
      initMemRefDescriptor<InputsElType>(inputs[i], inputSizes[i],
                                         initWithRandom, inputRange, i);

    for (int i = 0; i < NumOutputs; i++) {
      initMemRefDescriptor<OutputsElType>(outputs[i], outputSizes[i]);
      // set default result values to -1
      auto nEls = getNumElements(outputSizes[i]);
      std::fill((OutputsElType *)outputs[i].base,
                (OutputsElType *)outputs[i].base + nEls, -1);
    }
    if (verbose_) {
      printf("[%s v.%d] Allocated DRAM arrays (host VAs):", name_.c_str(),
             version_);
      for (int i = 0; i < NumInputs; i++)
        printf(" ANOOP input%d = %p ", i, inputs[i].base);
      for (int i = 0; i < NumOutputs; i++)
        printf(" ANOOP-1 output%d = %p ", i, outputs[i].base);
      printf("\n");
    }
  }

  template <typename ElType>
  int validateResult(size_t index, ElType *expected, bool printErrs = false,
                     float tolerance = 1e-5) {
    if (verbose_) {
      printf("[%s v.%d] Model executed successfully. Validating result...",
             name_.c_str(), version_);
    }

    int retCode = 0;
    size_t nEls = getNumElements(outputs[index].shape);
    float sqrSumOfDiff = 0.0;
    for (size_t j = 0; j < nEls; j++) {
      sqrSumOfDiff +=
          std::pow(((ElType *)outputs[index].base)[j] - expected[j], 2);
      if (std::abs(((ElType *)outputs[index].base)[j] - expected[j]) >
          tolerance) {
        retCode = 1;
        if (printErrs && j < MAX_RESULT_VALUES_TO_PRINT) {
          printf("Mismatch at index %d: expected %1.6f, got %1.6f\n", (int)j,
                 expected[j], ((ElType *)outputs[index].base)[j]);
        }
        if (retCode && j == MAX_RESULT_VALUES_TO_PRINT)
          printf("... (more mismatches not printed; maximum %d reached) ...\n",
                 MAX_RESULT_VALUES_TO_PRINT);
      }
    }
    // Compute the relative error: norm2(result) / norm2(expected)
    float sqrSumExpected = 0.0;
    for (size_t j = 0; j < nEls; j++)
      sqrSumExpected += std::pow(expected[j], 2);

    float relativeErr = std::sqrt(sqrSumOfDiff) / std::sqrt(sqrSumExpected);
    if (verbose_) {
      retCode ? printf("\n[%s v.%d] FAILED [relative err=%1.6f]\n",
                       name_.c_str(), version_, relativeErr)
              : printf("\n[%s v.%d] PASS [relative err=%1.6f]\n", name_.c_str(),
                       version_, relativeErr);
    }
    return retCode;
  }

  size_t getNumElements(const MemRefDescriptor<Rank> &memref) const {
    return getNumElements(memref.shape);
  }

  template <typename ElType>
  void writeToFile(void *data, size_t numElements,
                   const std::string &filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
      printf("[%s v.%d] Error opening file %s for writing.", name_.c_str(),
             version_, filename.c_str());
      return;
    }
    ofs.write((char *)data, numElements * sizeof(ElType));
    ofs.close();
  }

  template <typename ElType>
  void readFromFile(void *data, size_t numElements,
                    const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
      printf("[%s v.%d] Error opening file %s for reading.", name_.c_str(),
             version_, filename.c_str());
      return;
    }
    ifs.read((char *)data, numElements * sizeof(ElType));
    ifs.close();
  }

  std::string getName() const { return name_; }
  std::string getVersion() const { return std::to_string(version_); }

  MemRefDescriptor<Rank> inputs[NumInputs];
  MemRefDescriptor<Rank> outputs[NumOutputs];

private:
  std::string name_;
  int version_ = 1;
  bool verbose_ = false;

  template <typename ElType>
  void initMemRefDescriptor(MemRefDescriptor<Rank> &memref, size_t shape[Rank],
                            bool initWithRandom = false,
                            std::array<int, 2> inputRange = {-10, 10},
                            int seed = 42) {
    size_t nBytes = sizeof(ElType);
    for (int i = 0; i < Rank; i++) {
      nBytes *= shape[i];
    }
    memref.base = tsi_alloc(nBytes);
    memref.data = memref.base;
    memref.offset = 0;
    printf("\n checking Shape value %d \n\n", memref.shape[0]);
#if 0
    for (int i = 0; i < Rank; i++) {
      memref.shape[i] = shape[i];
      memref.strides[i] = 1;
      for (int j = i + 1; j < Rank; j++) {
        memref.strides[i] *= shape[j];
      }
    }
 #endif
    if (initWithRandom) {
      std::mt19937 gen(seed); // fixed seed
      std::uniform_real_distribution<float> dist(inputRange[0], inputRange[1]);
      for (size_t i = 0; i < getNumElements(shape); i++) {
        static_cast<ElType *>(memref.data)[i] = static_cast<ElType>(dist(gen));
      }
    }
  }

  size_t getNumElements(const int64_t shape[Rank]) const {
    size_t numElements = 1;
    printf("\n Anoop Rank %d and shape[Rank] %d \n\n", Rank, shape[Rank]);
    for (int i = 0; i < Rank; i++) {
      numElements *= shape[i];
    }
    printf("\n numElements %d \n", numElements);
    return numElements;
  }

  size_t getNumElements(const size_t shape[Rank]) const {
    return getNumElements(reinterpret_cast<const int64_t *>(shape));
  }
};
