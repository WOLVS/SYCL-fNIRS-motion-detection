# CMake generated Testfile for 
# Source directory: /home/yunyi/Desktop/zyy-workspace/oneDNN/tests
# Build directory: /home/yunyi/Desktop/zyy-workspace/oneDNN/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(api-c "api-c")
set_tests_properties(api-c PROPERTIES  _BACKTRACE_TRIPLES "/home/yunyi/Desktop/zyy-workspace/oneDNN/cmake/utils.cmake;44;add_test;/home/yunyi/Desktop/zyy-workspace/oneDNN/cmake/utils.cmake;56;add_dnnl_test;/home/yunyi/Desktop/zyy-workspace/oneDNN/tests/CMakeLists.txt;73;register_exe;/home/yunyi/Desktop/zyy-workspace/oneDNN/tests/CMakeLists.txt;0;")
add_test(test_c_symbols-c "test_c_symbols-c")
set_tests_properties(test_c_symbols-c PROPERTIES  _BACKTRACE_TRIPLES "/home/yunyi/Desktop/zyy-workspace/oneDNN/cmake/utils.cmake;44;add_test;/home/yunyi/Desktop/zyy-workspace/oneDNN/cmake/utils.cmake;56;add_dnnl_test;/home/yunyi/Desktop/zyy-workspace/oneDNN/tests/CMakeLists.txt;84;register_exe;/home/yunyi/Desktop/zyy-workspace/oneDNN/tests/CMakeLists.txt;0;")
subdirs("gtests")
subdirs("benchdnn")
subdirs("noexcept")
