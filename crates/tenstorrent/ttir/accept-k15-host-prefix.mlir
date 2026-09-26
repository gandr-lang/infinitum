module {
  func.func @accept(%arg0: tensor<16x970x256xbf16>, %arg1: tensor<16x970x256xbf16>, %arg2: tensor<16x970x256xbf16>, %arg3: tensor<16x970xbf16>, %arg4: tensor<16x970xbf16>) -> (tensor<16x1xbf16>, tensor<16x1xbf16>, tensor<16x1xbf16>) {
    %0 = "ttir.full"() <{fill_value = 0xFF800000 : f32, shape = array<i32: 16, 970, 256>}> : () -> tensor<16x970x256xbf16>
    %1 = "ttir.where"(%arg1, %0, %arg0) : (tensor<16x970x256xbf16>, tensor<16x970x256xbf16>, tensor<16x970x256xbf16>) -> tensor<16x970x256xbf16>
    %2 = "ttir.max"(%1) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<16x970x256xbf16>) -> tensor<16x970x1xbf16>
    %3 = "ttir.max"(%1) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<16x970x256xbf16>) -> tensor<16x970xbf16>
    %4 = "ttir.broadcast"(%2) <{broadcast_dimensions = array<i64: 1, 1, 256>}> : (tensor<16x970x1xbf16>) -> tensor<16x970x256xbf16>
    %5 = "ttir.eq"(%1, %4) : (tensor<16x970x256xbf16>, tensor<16x970x256xbf16>) -> tensor<16x970x256xbf16>
    %6 = "ttir.full"() <{fill_value = 1.024000e+03 : f32, shape = array<i32: 16, 970, 256>}> : () -> tensor<16x970x256xbf16>
    %7 = "ttir.where"(%5, %arg2, %6) : (tensor<16x970x256xbf16>, tensor<16x970x256xbf16>, tensor<16x970x256xbf16>) -> tensor<16x970x256xbf16>
    %8 = "ttir.min"(%7) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<16x970x256xbf16>) -> tensor<16x970xbf16>
    %9 = "ttir.max"(%3) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<16x970xbf16>) -> tensor<16x1xbf16>
    %10 = "ttir.broadcast"(%9) <{broadcast_dimensions = array<i64: 1, 970>}> : (tensor<16x1xbf16>) -> tensor<16x970xbf16>
    %11 = "ttir.eq"(%3, %10) : (tensor<16x970xbf16>, tensor<16x970xbf16>) -> tensor<16x970xbf16>
    %12 = "ttir.full"() <{fill_value = 1.024000e+03 : f32, shape = array<i32: 16, 970>}> : () -> tensor<16x970xbf16>
    %13 = "ttir.where"(%11, %arg3, %12) : (tensor<16x970xbf16>, tensor<16x970xbf16>, tensor<16x970xbf16>) -> tensor<16x970xbf16>
    %14 = "ttir.min"(%13) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<16x970xbf16>) -> tensor<16x1xbf16>
    %15 = "ttir.broadcast"(%14) <{broadcast_dimensions = array<i64: 1, 970>}> : (tensor<16x1xbf16>) -> tensor<16x970xbf16>
    %16 = "ttir.eq"(%arg3, %15) : (tensor<16x970xbf16>, tensor<16x970xbf16>) -> tensor<16x970xbf16>
    %17 = "ttir.logical_and"(%11, %16) : (tensor<16x970xbf16>, tensor<16x970xbf16>) -> tensor<16x970xbf16>
    %18 = "ttir.full"() <{fill_value = 1.024000e+03 : f32, shape = array<i32: 16, 970>}> : () -> tensor<16x970xbf16>
    %19 = "ttir.where"(%17, %arg4, %18) : (tensor<16x970xbf16>, tensor<16x970xbf16>, tensor<16x970xbf16>) -> tensor<16x970xbf16>
    %20 = "ttir.min"(%19) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<16x970xbf16>) -> tensor<16x1xbf16>
    %21 = "ttir.broadcast"(%20) <{broadcast_dimensions = array<i64: 1, 970>}> : (tensor<16x1xbf16>) -> tensor<16x970xbf16>
    %22 = "ttir.eq"(%arg4, %21) : (tensor<16x970xbf16>, tensor<16x970xbf16>) -> tensor<16x970xbf16>
    %23 = "ttir.logical_and"(%17, %22) : (tensor<16x970xbf16>, tensor<16x970xbf16>) -> tensor<16x970xbf16>
    %24 = "ttir.full"() <{fill_value = 1.024000e+03 : f32, shape = array<i32: 16, 970>}> : () -> tensor<16x970xbf16>
    %25 = "ttir.where"(%23, %8, %24) : (tensor<16x970xbf16>, tensor<16x970xbf16>, tensor<16x970xbf16>) -> tensor<16x970xbf16>
    %26 = "ttir.min"(%25) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<16x970xbf16>) -> tensor<16x1xbf16>
    return %14, %20, %26 : tensor<16x1xbf16>, tensor<16x1xbf16>, tensor<16x1xbf16>
  }
}
