#ifndef REINTERPRET_FWD_I32_GAUDI2_TEST_HPP
#define REINTERPRET_FWD_I32_GAUDI2_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "reinterpret_fwd_i32.hpp"

class ReinterpretFwdI32Gaudi2Test : public TestBase
{
public:
    ReinterpretFwdI32Gaudi2Test() {}
    ~ReinterpretFwdI32Gaudi2Test() {}
    int runTest();

    static void reinterpret_reference_implementation(
         const test::Tensor<float, 5>& input,
         test::Tensor<int32_t, 5>& output,
         const IndexSpace& indexSpace);
private:
    ReinterpretFwdI32Gaudi2Test(const ReinterpretFwdI32Gaudi2Test& other) = delete;
    ReinterpretFwdI32Gaudi2Test& operator=(const ReinterpretFwdI32Gaudi2Test& other) = delete;

};


#endif /* REINTERPRET_FWD_I32_GAUDI2_TEST_HPP */
