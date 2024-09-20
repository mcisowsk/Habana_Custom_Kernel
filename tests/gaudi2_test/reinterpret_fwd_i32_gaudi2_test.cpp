/**********************************************************************
Copyright (c) 2021 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "tensor.h"
#include "reinterpret_fwd_i32_gaudi2_test.hpp"
#include "entry_points.hpp"


void ReinterpretFwdI32Gaudi2Test::reinterpret_reference_implementation(
        const test::Tensor<float, 5>& input,
        test::Tensor<int32_t, 5>& output,
        const IndexSpace& indexSpace)
{

    int coords[5] = { 0 };
    for (int d0 = 0; d0 < indexSpace.size[0]; d0 += 1)
    {
        coords[0] = d0;
        for (int d1 = 0; d1 < indexSpace.size[1]; d1 += 1)
        {
            coords[1] = d1;
            for (int d2 = 0; d2 < indexSpace.size[2]; d2 += 1)
            {
                coords[2] = d2;
                for (int d3 = 0; d3 < indexSpace.size[3]; d3 += 1)
                {
                    coords[3] = d3;
                    for (int d4 = 0; d4 < indexSpace.size[4]; d4 += 1)
                    {
                        coords[4] = d4;
                        float tmp = input.ElementAt(coords);
                        output.SetElement(coords, *reinterpret_cast<int32_t*>(&tmp));
                    }
                }
            }
        }
    }
}

int ReinterpretFwdI32Gaudi2Test::runTest()
{
    // Initialize input data
    const int fm_dim0 = 64;
    const int fm_dim1 = 4;
    const int fm_dim2 = 9;
    const int fm_dim3 = 3;
    const int fm_dim4 = 2;

    uint64_t fmInitializer[] = {fm_dim0, fm_dim1, fm_dim2, fm_dim3, fm_dim4};
    float_5DTensor input(fmInitializer);
    input.FillWithValueFloat(3.234);

    int32_5DTensor output(fmInitializer);
    int32_5DTensor output_ref(fmInitializer);

    IndexSpace index_space = {{0}};
    index_space.size[0] = fm_dim0;
    index_space.size[1] = fm_dim1;
    index_space.size[2] = fm_dim2;
    index_space.size[3] = fm_dim3;
    index_space.size[4] = fm_dim4;

    // execute reference implementation of the kernel.
    reinterpret_reference_implementation(input,
                                     output_ref,
                                     index_space);

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

    tpc_lib_api::GuidInfo *guids = nullptr;
    unsigned kernelCount = 0;
    tpc_lib_api::GlueCodeReturn result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    guids = new tpc_lib_api::GuidInfo[kernelCount];
    result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_REINTERPRET_FWD_I32].name);
    result  = InstantiateTpcKernel(&m_in_defs, &m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!!" << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    input.Print(0);
    output.Print(0);
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element])  > 1e-8)
        {
            std::cout << "Reinterpret FWD I32 Gaudi 2 test failed!!" << std::endl;
            ReleaseKernelNames(guids, kernelCount);
            return -1;
        }
    }

    std::cout << "Reinterpret FWD I32 Gaudi 2 test pass!!" << std::endl;

    return 0;
}
