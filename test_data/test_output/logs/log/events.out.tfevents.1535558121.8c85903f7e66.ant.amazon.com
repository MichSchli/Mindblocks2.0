       �K"	  @z���Abrain.Event:2�3�h      jr�t	|�Az���A"��
�
abc_W0/initial_valueConst*�
value�B�"�����>�P������r�>V��I��>�Ѿcھ�ԣ� {뽑/�>(�����>��<��-=�>o>l!Ծ����j>��˽p�~���8�v���F�>#z�>��*��=�o=�?ž�=�Ź�@�>�y>���<#J�>�ʾK�z��n>K�>�N�>�,�>vS��5h�n]�����>���=(&��Ep��"�ݽG���S ��N>�s�>ι���º<N�a��b��2���O�>7���v���!��>.����p����>��>���v�>$7
>��u�������>�ֲ>�)�>�����M=�&�>���= �*
dtype0*
_output_shapes

:
z
abc_W0
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
abc_W0/AssignAssignabc_W0abc_W0/initial_value*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@abc_W0
c
abc_W0/readIdentityabc_W0*
T0*
_class
loc:@abc_W0*
_output_shapes

:
�
abc_b0/initial_valueConst*e
value\BZ"P                                                                                *
dtype0*
_output_shapes
:
r
abc_b0
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
abc_b0/AssignAssignabc_b0abc_b0/initial_value*
use_locking(*
T0*
_class
loc:@abc_b0*
validate_shape(*
_output_shapes
:
_
abc_b0/readIdentityabc_b0*
T0*
_class
loc:@abc_b0*
_output_shapes
:
�
abc_W1/initial_valueConst*�
value�B�"���$.漪i�>�@t��S>��Q�-�7>o����k�=�A->��>�b�����>�8�=��B���b>`>4ټ«�����>b�D�"�������>�K"=�#>�����8�=+m�=��=���=:Z>�ީ�~� �z�y0�>C%�>��>��>���>�D����>j�>Y\�>���>���{�߽�9��lS��6�=�@�>��'�e�>�Y�>*�>/�]>���>^�>PU���z�=*
dtype0*
_output_shapes

:
z
abc_W1
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
abc_W1/AssignAssignabc_W1abc_W1/initial_value*
use_locking(*
T0*
_class
loc:@abc_W1*
validate_shape(*
_output_shapes

:
c
abc_W1/readIdentityabc_W1*
T0*
_class
loc:@abc_W1*
_output_shapes

:
i
abc_b1/initial_valueConst*!
valueB"            *
dtype0*
_output_shapes
:
r
abc_b1
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
abc_b1/AssignAssignabc_b1abc_b1/initial_value*
use_locking(*
T0*
_class
loc:@abc_b1*
validate_shape(*
_output_shapes
:
_
abc_b1/readIdentityabc_b1*
T0*
_class
loc:@abc_b1*
_output_shapes
:
n
PlaceholderPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
MatMulMatMulPlaceholder_1abc_W0/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
Q
addAddMatMulabc_b0/read*
T0*'
_output_shapes
:���������
C
ReluReluadd*
T0*'
_output_shapes
:���������
}
MatMul_1MatMulReluabc_W1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
U
add_1AddMatMul_1abc_b1/read*
T0*'
_output_shapes
:���������
V
SqueezeSqueezePlaceholder*
squeeze_dims
 *
T0*
_output_shapes
:
y
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapeSqueeze*
T0*
out_type0*#
_output_shapes
:���������
�
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsadd_1Squeeze*
T0*6
_output_shapes$
":���������:���������*
Tlabels0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
gradients/Mean_grad/ShapeShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
gradients/Mean_grad/Shape_1ShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
�
gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:���������*
T0
�
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*'
_output_shapes
:���������
�
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*'
_output_shapes
:���������*
T0
b
gradients/add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*gradients/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_1_grad/Sum_1SumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:���������
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyabc_W1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0*'
_output_shapes
:���������
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyabc_W0/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder_1+gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:
�
global_norm/L2LossL2Loss0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
�
global_norm/L2Loss_1L2Loss-gradients/add_grad/tuple/control_dependency_1*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: 
�
global_norm/L2Loss_2L2Loss2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_3L2Loss/gradients/add_1_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
: 
�
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3*
T0*

axis *
N*
_output_shapes
:
[
global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
global_norm/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
clip_by_global_norm/truediv_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
_output_shapes
: *
T0
�
clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
^
clip_by_global_norm/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
_output_shapes
: *
T0
�
clip_by_global_norm/mul_1Mul0gradients/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:
�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:
�
clip_by_global_norm/mul_2Mul-gradients/add_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
�
clip_by_global_norm/mul_3Mul2gradients/MatMul_1_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
_output_shapes

:*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
�
clip_by_global_norm/mul_4Mul/gradients/add_1_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:
y
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@abc_W0*
valueB
 *fff?
�
beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@abc_W0*
	container 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes
: *
use_locking(
e
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@abc_W0*
_output_shapes
: 
y
beta2_power/initial_valueConst*
_class
loc:@abc_W0*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@abc_W0
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@abc_W0
e
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@abc_W0*
_output_shapes
: 
�
abc_W0/Adam/Initializer/zerosConst*
_class
loc:@abc_W0*
valueB*    *
dtype0*
_output_shapes

:
�
abc_W0/Adam
VariableV2*
shared_name *
_class
loc:@abc_W0*
	container *
shape
:*
dtype0*
_output_shapes

:
�
abc_W0/Adam/AssignAssignabc_W0/Adamabc_W0/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes

:
m
abc_W0/Adam/readIdentityabc_W0/Adam*
_output_shapes

:*
T0*
_class
loc:@abc_W0
�
abc_W0/Adam_1/Initializer/zerosConst*
_class
loc:@abc_W0*
valueB*    *
dtype0*
_output_shapes

:
�
abc_W0/Adam_1
VariableV2*
shared_name *
_class
loc:@abc_W0*
	container *
shape
:*
dtype0*
_output_shapes

:
�
abc_W0/Adam_1/AssignAssignabc_W0/Adam_1abc_W0/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes

:
q
abc_W0/Adam_1/readIdentityabc_W0/Adam_1*
T0*
_class
loc:@abc_W0*
_output_shapes

:
�
abc_b0/Adam/Initializer/zerosConst*
_class
loc:@abc_b0*
valueB*    *
dtype0*
_output_shapes
:
�
abc_b0/Adam
VariableV2*
shared_name *
_class
loc:@abc_b0*
	container *
shape:*
dtype0*
_output_shapes
:
�
abc_b0/Adam/AssignAssignabc_b0/Adamabc_b0/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@abc_b0
i
abc_b0/Adam/readIdentityabc_b0/Adam*
T0*
_class
loc:@abc_b0*
_output_shapes
:
�
abc_b0/Adam_1/Initializer/zerosConst*
_class
loc:@abc_b0*
valueB*    *
dtype0*
_output_shapes
:
�
abc_b0/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@abc_b0*
	container *
shape:
�
abc_b0/Adam_1/AssignAssignabc_b0/Adam_1abc_b0/Adam_1/Initializer/zeros*
T0*
_class
loc:@abc_b0*
validate_shape(*
_output_shapes
:*
use_locking(
m
abc_b0/Adam_1/readIdentityabc_b0/Adam_1*
_output_shapes
:*
T0*
_class
loc:@abc_b0
�
abc_W1/Adam/Initializer/zerosConst*
_class
loc:@abc_W1*
valueB*    *
dtype0*
_output_shapes

:
�
abc_W1/Adam
VariableV2*
shared_name *
_class
loc:@abc_W1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
abc_W1/Adam/AssignAssignabc_W1/Adamabc_W1/Adam/Initializer/zeros*
T0*
_class
loc:@abc_W1*
validate_shape(*
_output_shapes

:*
use_locking(
m
abc_W1/Adam/readIdentityabc_W1/Adam*
_output_shapes

:*
T0*
_class
loc:@abc_W1
�
abc_W1/Adam_1/Initializer/zerosConst*
_class
loc:@abc_W1*
valueB*    *
dtype0*
_output_shapes

:
�
abc_W1/Adam_1
VariableV2*
shared_name *
_class
loc:@abc_W1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
abc_W1/Adam_1/AssignAssignabc_W1/Adam_1abc_W1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@abc_W1*
validate_shape(*
_output_shapes

:
q
abc_W1/Adam_1/readIdentityabc_W1/Adam_1*
T0*
_class
loc:@abc_W1*
_output_shapes

:
�
abc_b1/Adam/Initializer/zerosConst*
_class
loc:@abc_b1*
valueB*    *
dtype0*
_output_shapes
:
�
abc_b1/Adam
VariableV2*
shared_name *
_class
loc:@abc_b1*
	container *
shape:*
dtype0*
_output_shapes
:
�
abc_b1/Adam/AssignAssignabc_b1/Adamabc_b1/Adam/Initializer/zeros*
T0*
_class
loc:@abc_b1*
validate_shape(*
_output_shapes
:*
use_locking(
i
abc_b1/Adam/readIdentityabc_b1/Adam*
_output_shapes
:*
T0*
_class
loc:@abc_b1
�
abc_b1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@abc_b1*
valueB*    
�
abc_b1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@abc_b1*
	container *
shape:
�
abc_b1/Adam_1/AssignAssignabc_b1/Adam_1abc_b1/Adam_1/Initializer/zeros*
T0*
_class
loc:@abc_b1*
validate_shape(*
_output_shapes
:*
use_locking(
m
abc_b1/Adam_1/readIdentityabc_b1/Adam_1*
T0*
_class
loc:@abc_b1*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Adam/update_abc_W0/ApplyAdam	ApplyAdamabc_W0abc_W0/Adamabc_W0/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*
_class
loc:@abc_W0
�
Adam/update_abc_b0/ApplyAdam	ApplyAdamabc_b0abc_b0/Adamabc_b0/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0*
_class
loc:@abc_b0*
use_nesterov( *
_output_shapes
:
�
Adam/update_abc_W1/ApplyAdam	ApplyAdamabc_W1abc_W1/Adamabc_W1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
T0*
_class
loc:@abc_W1*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
Adam/update_abc_b1/ApplyAdam	ApplyAdamabc_b1abc_b1/Adamabc_b1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*
_class
loc:@abc_b1
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_abc_W0/ApplyAdam^Adam/update_abc_W1/ApplyAdam^Adam/update_abc_b0/ApplyAdam^Adam/update_abc_b1/ApplyAdam*
T0*
_class
loc:@abc_W0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes
: *
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_abc_W0/ApplyAdam^Adam/update_abc_W1/ApplyAdam^Adam/update_abc_b0/ApplyAdam^Adam/update_abc_b1/ApplyAdam*
T0*
_class
loc:@abc_W0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes
: 
�
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_abc_W0/ApplyAdam^Adam/update_abc_W1/ApplyAdam^Adam/update_abc_b0/ApplyAdam^Adam/update_abc_b1/ApplyAdam
p
Placeholder_2Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
MatMul_2MatMulPlaceholder_2abc_W0/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
U
add_2AddMatMul_2abc_b0/read*'
_output_shapes
:���������*
T0
G
Relu_1Reluadd_2*
T0*'
_output_shapes
:���������

MatMul_3MatMulRelu_1abc_W1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
U
add_3AddMatMul_3abc_b1/read*
T0*'
_output_shapes
:���������
K
SoftmaxSoftmaxadd_3*
T0*'
_output_shapes
:���������
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
MatMul_4MatMulPlaceholder_3abc_W0/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
U
add_4AddMatMul_4abc_b0/read*
T0*'
_output_shapes
:���������
G
Relu_2Reluadd_4*'
_output_shapes
:���������*
T0

MatMul_5MatMulRelu_2abc_W1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
U
add_5AddMatMul_5abc_b1/read*
T0*'
_output_shapes
:���������
M
	Softmax_1Softmaxadd_5*
T0*'
_output_shapes
:���������
�
initNoOp^abc_W0/Adam/Assign^abc_W0/Adam_1/Assign^abc_W0/Assign^abc_W1/Adam/Assign^abc_W1/Adam_1/Assign^abc_W1/Assign^abc_b0/Adam/Assign^abc_b0/Adam_1/Assign^abc_b0/Assign^abc_b1/Adam/Assign^abc_b1/Adam_1/Assign^abc_b1/Assign^beta1_power/Assign^beta2_power/Assign"�;(Jgz      �4�)	ƉBz���AJ��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.10.02v1.10.0-rc1-19-g656e7a2b34��
�
abc_W0/initial_valueConst*�
value�B�"�����>�P������r�>V��I��>�Ѿcھ�ԣ� {뽑/�>(�����>��<��-=�>o>l!Ծ����j>��˽p�~���8�v���F�>#z�>��*��=�o=�?ž�=�Ź�@�>�y>���<#J�>�ʾK�z��n>K�>�N�>�,�>vS��5h�n]�����>���=(&��Ep��"�ݽG���S ��N>�s�>ι���º<N�a��b��2���O�>7���v���!��>.����p����>��>���v�>$7
>��u�������>�ֲ>�)�>�����M=�&�>���= �*
dtype0*
_output_shapes

:
z
abc_W0
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
abc_W0/AssignAssignabc_W0abc_W0/initial_value*
use_locking(*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes

:
c
abc_W0/readIdentityabc_W0*
_output_shapes

:*
T0*
_class
loc:@abc_W0
�
abc_b0/initial_valueConst*
dtype0*
_output_shapes
:*e
value\BZ"P                                                                                
r
abc_b0
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
abc_b0/AssignAssignabc_b0abc_b0/initial_value*
use_locking(*
T0*
_class
loc:@abc_b0*
validate_shape(*
_output_shapes
:
_
abc_b0/readIdentityabc_b0*
_output_shapes
:*
T0*
_class
loc:@abc_b0
�
abc_W1/initial_valueConst*
dtype0*
_output_shapes

:*�
value�B�"���$.漪i�>�@t��S>��Q�-�7>o����k�=�A->��>�b�����>�8�=��B���b>`>4ټ«�����>b�D�"�������>�K"=�#>�����8�=+m�=��=���=:Z>�ީ�~� �z�y0�>C%�>��>��>���>�D����>j�>Y\�>���>���{�߽�9��lS��6�=�@�>��'�e�>�Y�>*�>/�]>���>^�>PU���z�=
z
abc_W1
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
abc_W1/AssignAssignabc_W1abc_W1/initial_value*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@abc_W1
c
abc_W1/readIdentityabc_W1*
T0*
_class
loc:@abc_W1*
_output_shapes

:
i
abc_b1/initial_valueConst*!
valueB"            *
dtype0*
_output_shapes
:
r
abc_b1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
abc_b1/AssignAssignabc_b1abc_b1/initial_value*
use_locking(*
T0*
_class
loc:@abc_b1*
validate_shape(*
_output_shapes
:
_
abc_b1/readIdentityabc_b1*
_output_shapes
:*
T0*
_class
loc:@abc_b1
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
MatMulMatMulPlaceholder_1abc_W0/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
Q
addAddMatMulabc_b0/read*
T0*'
_output_shapes
:���������
C
ReluReluadd*'
_output_shapes
:���������*
T0
}
MatMul_1MatMulReluabc_W1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
U
add_1AddMatMul_1abc_b1/read*
T0*'
_output_shapes
:���������
V
SqueezeSqueezePlaceholder*
squeeze_dims
 *
T0*
_output_shapes
:
y
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapeSqueeze*#
_output_shapes
:���������*
T0*
out_type0
�
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsadd_1Squeeze*6
_output_shapes$
":���������:���������*
Tlabels0*
T0
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
gradients/Mean_grad/ShapeShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
�
gradients/Mean_grad/Shape_1ShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:���������
�
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*'
_output_shapes
:���������
�
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:���������
b
gradients/add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*gradients/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_1_grad/Sum_1SumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:���������
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyabc_W1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:���������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:���������*
T0
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyabc_W0/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder_1+gradients/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
�
global_norm/L2LossL2Loss0gradients/MatMul_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_1L2Loss-gradients/add_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
global_norm/L2Loss_2L2Loss2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_3L2Loss/gradients/add_1_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
: 
�
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3*
T0*

axis *
N*
_output_shapes
:
[
global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
global_norm/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   @
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
_output_shapes
: *
T0
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
clip_by_global_norm/truediv_1/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 
�
clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
^
clip_by_global_norm/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
T0*
_output_shapes
: 
�
clip_by_global_norm/mul_1Mul0gradients/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
_output_shapes

:*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
_output_shapes

:*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
�
clip_by_global_norm/mul_2Mul-gradients/add_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
�
clip_by_global_norm/mul_3Mul2gradients/MatMul_1_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
clip_by_global_norm/mul_4Mul/gradients/add_1_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:
y
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@abc_W0*
valueB
 *fff?
�
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@abc_W0*
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes
: *
use_locking(
e
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@abc_W0
y
beta2_power/initial_valueConst*
_class
loc:@abc_W0*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shared_name *
_class
loc:@abc_W0*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes
: 
e
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@abc_W0
�
abc_W0/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*
_class
loc:@abc_W0*
valueB*    
�
abc_W0/Adam
VariableV2*
_class
loc:@abc_W0*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
abc_W0/Adam/AssignAssignabc_W0/Adamabc_W0/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes

:
m
abc_W0/Adam/readIdentityabc_W0/Adam*
T0*
_class
loc:@abc_W0*
_output_shapes

:
�
abc_W0/Adam_1/Initializer/zerosConst*
_class
loc:@abc_W0*
valueB*    *
dtype0*
_output_shapes

:
�
abc_W0/Adam_1
VariableV2*
_class
loc:@abc_W0*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
abc_W0/Adam_1/AssignAssignabc_W0/Adam_1abc_W0/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes

:
q
abc_W0/Adam_1/readIdentityabc_W0/Adam_1*
T0*
_class
loc:@abc_W0*
_output_shapes

:
�
abc_b0/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@abc_b0*
valueB*    
�
abc_b0/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@abc_b0*
	container 
�
abc_b0/Adam/AssignAssignabc_b0/Adamabc_b0/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@abc_b0
i
abc_b0/Adam/readIdentityabc_b0/Adam*
T0*
_class
loc:@abc_b0*
_output_shapes
:
�
abc_b0/Adam_1/Initializer/zerosConst*
_class
loc:@abc_b0*
valueB*    *
dtype0*
_output_shapes
:
�
abc_b0/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@abc_b0*
	container *
shape:
�
abc_b0/Adam_1/AssignAssignabc_b0/Adam_1abc_b0/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@abc_b0*
validate_shape(*
_output_shapes
:
m
abc_b0/Adam_1/readIdentityabc_b0/Adam_1*
T0*
_class
loc:@abc_b0*
_output_shapes
:
�
abc_W1/Adam/Initializer/zerosConst*
_class
loc:@abc_W1*
valueB*    *
dtype0*
_output_shapes

:
�
abc_W1/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@abc_W1*
	container *
shape
:
�
abc_W1/Adam/AssignAssignabc_W1/Adamabc_W1/Adam/Initializer/zeros*
T0*
_class
loc:@abc_W1*
validate_shape(*
_output_shapes

:*
use_locking(
m
abc_W1/Adam/readIdentityabc_W1/Adam*
T0*
_class
loc:@abc_W1*
_output_shapes

:
�
abc_W1/Adam_1/Initializer/zerosConst*
_class
loc:@abc_W1*
valueB*    *
dtype0*
_output_shapes

:
�
abc_W1/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@abc_W1*
	container *
shape
:
�
abc_W1/Adam_1/AssignAssignabc_W1/Adam_1abc_W1/Adam_1/Initializer/zeros*
T0*
_class
loc:@abc_W1*
validate_shape(*
_output_shapes

:*
use_locking(
q
abc_W1/Adam_1/readIdentityabc_W1/Adam_1*
_output_shapes

:*
T0*
_class
loc:@abc_W1
�
abc_b1/Adam/Initializer/zerosConst*
_class
loc:@abc_b1*
valueB*    *
dtype0*
_output_shapes
:
�
abc_b1/Adam
VariableV2*
shared_name *
_class
loc:@abc_b1*
	container *
shape:*
dtype0*
_output_shapes
:
�
abc_b1/Adam/AssignAssignabc_b1/Adamabc_b1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@abc_b1*
validate_shape(*
_output_shapes
:
i
abc_b1/Adam/readIdentityabc_b1/Adam*
T0*
_class
loc:@abc_b1*
_output_shapes
:
�
abc_b1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@abc_b1*
valueB*    
�
abc_b1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@abc_b1*
	container *
shape:
�
abc_b1/Adam_1/AssignAssignabc_b1/Adam_1abc_b1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@abc_b1*
validate_shape(*
_output_shapes
:
m
abc_b1/Adam_1/readIdentityabc_b1/Adam_1*
_output_shapes
:*
T0*
_class
loc:@abc_b1
W
Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Adam/update_abc_W0/ApplyAdam	ApplyAdamabc_W0abc_W0/Adamabc_W0/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_locking( *
T0*
_class
loc:@abc_W0*
use_nesterov( *
_output_shapes

:
�
Adam/update_abc_b0/ApplyAdam	ApplyAdamabc_b0abc_b0/Adamabc_b0/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0*
_class
loc:@abc_b0*
use_nesterov( *
_output_shapes
:
�
Adam/update_abc_W1/ApplyAdam	ApplyAdamabc_W1abc_W1/Adamabc_W1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
T0*
_class
loc:@abc_W1*
use_nesterov( *
_output_shapes

:
�
Adam/update_abc_b1/ApplyAdam	ApplyAdamabc_b1abc_b1/Adamabc_b1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_locking( *
T0*
_class
loc:@abc_b1*
use_nesterov( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_abc_W0/ApplyAdam^Adam/update_abc_W1/ApplyAdam^Adam/update_abc_b0/ApplyAdam^Adam/update_abc_b1/ApplyAdam*
T0*
_class
loc:@abc_W0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_abc_W0/ApplyAdam^Adam/update_abc_W1/ApplyAdam^Adam/update_abc_b0/ApplyAdam^Adam/update_abc_b1/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@abc_W0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@abc_W0*
validate_shape(*
_output_shapes
: 
�
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_abc_W0/ApplyAdam^Adam/update_abc_W1/ApplyAdam^Adam/update_abc_b0/ApplyAdam^Adam/update_abc_b1/ApplyAdam
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
MatMul_2MatMulPlaceholder_2abc_W0/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
U
add_2AddMatMul_2abc_b0/read*'
_output_shapes
:���������*
T0
G
Relu_1Reluadd_2*'
_output_shapes
:���������*
T0

MatMul_3MatMulRelu_1abc_W1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
U
add_3AddMatMul_3abc_b1/read*
T0*'
_output_shapes
:���������
K
SoftmaxSoftmaxadd_3*
T0*'
_output_shapes
:���������
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
MatMul_4MatMulPlaceholder_3abc_W0/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
U
add_4AddMatMul_4abc_b0/read*'
_output_shapes
:���������*
T0
G
Relu_2Reluadd_4*'
_output_shapes
:���������*
T0

MatMul_5MatMulRelu_2abc_W1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
U
add_5AddMatMul_5abc_b1/read*
T0*'
_output_shapes
:���������
M
	Softmax_1Softmaxadd_5*'
_output_shapes
:���������*
T0
�
initNoOp^abc_W0/Adam/Assign^abc_W0/Adam_1/Assign^abc_W0/Assign^abc_W1/Adam/Assign^abc_W1/Adam_1/Assign^abc_W1/Assign^abc_b0/Adam/Assign^abc_b0/Adam_1/Assign^abc_b0/Assign^abc_b1/Adam/Assign^abc_b1/Adam_1/Assign^abc_b1/Assign^beta1_power/Assign^beta2_power/Assign""�
trainable_variables��
B
abc_W0:0abc_W0/Assignabc_W0/read:02abc_W0/initial_value:08
B
abc_b0:0abc_b0/Assignabc_b0/read:02abc_b0/initial_value:08
B
abc_W1:0abc_W1/Assignabc_W1/read:02abc_W1/initial_value:08
B
abc_b1:0abc_b1/Assignabc_b1/read:02abc_b1/initial_value:08"
train_op

Adam"�	
	variables�	�	
B
abc_W0:0abc_W0/Assignabc_W0/read:02abc_W0/initial_value:08
B
abc_b0:0abc_b0/Assignabc_b0/read:02abc_b0/initial_value:08
B
abc_W1:0abc_W1/Assignabc_W1/read:02abc_W1/initial_value:08
B
abc_b1:0abc_b1/Assignabc_b1/read:02abc_b1/initial_value:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
X
abc_W0/Adam:0abc_W0/Adam/Assignabc_W0/Adam/read:02abc_W0/Adam/Initializer/zeros:0
`
abc_W0/Adam_1:0abc_W0/Adam_1/Assignabc_W0/Adam_1/read:02!abc_W0/Adam_1/Initializer/zeros:0
X
abc_b0/Adam:0abc_b0/Adam/Assignabc_b0/Adam/read:02abc_b0/Adam/Initializer/zeros:0
`
abc_b0/Adam_1:0abc_b0/Adam_1/Assignabc_b0/Adam_1/read:02!abc_b0/Adam_1/Initializer/zeros:0
X
abc_W1/Adam:0abc_W1/Adam/Assignabc_W1/Adam/read:02abc_W1/Adam/Initializer/zeros:0
`
abc_W1/Adam_1:0abc_W1/Adam_1/Assignabc_W1/Adam_1/read:02!abc_W1/Adam_1/Initializer/zeros:0
X
abc_b1/Adam:0abc_b1/Adam/Assignabc_b1/Adam/read:02abc_b1/Adam/Initializer/zeros:0
`
abc_b1/Adam_1:0abc_b1/Adam_1/Assignabc_b1/Adam_1/read:02!abc_b1/Adam_1/Initializer/zeros:0-�:�>      _z��	��Bz���AB�}
mySess�}
�}
�}
,/job:localhost/replica:0/task:0/device:CPU:05
_SOURCE����ϒ� (B_SOURCE = NoOp()H����ϒ�b e
Adam/learning_rateɱ��ϒ� (
: "cpu0�ב���BAdam/learning_rate = Const()H����ϒ�bl
abc_W0/Adam_1˱��ϒ� (: "�cpu0����Babc_W0/Adam_1 = VariableV2()H����ϒ�Pb�j
abc_W1/Adam_1ױ��ϒ� (: "�cpu0����Babc_W1/Adam_1 = VariableV2()H����ϒ�b�j
abc_W0/AdamƱ��ϒ� (: "�cpu0������Babc_W0/Adam = VariableV2()H����ϒ�Pb�`
abc_b0/Adamݱ��ϒ� (	:"Pcpu0��Ȉ��Babc_b0/Adam = VariableV2()H����ϒ�bPh
abc_W1/Adam˱��ϒ� (: "�cpu0������Babc_W1/Adam = VariableV2()H����ϒ�Pb�d
abc_b0/Adam_1���ϒ� (:"Pcpu0���Babc_b0/Adam_1 = VariableV2()H����ϒ�bPb
abc_b1/Adam���ϒ� (:"cpu0������Babc_b1/Adam = VariableV2()H����ϒ�Pb\
beta1_power���ϒ� (: "cpu0����Bbeta1_power = VariableV2()H����ϒ�bf
abc_b1/Adam_1���ϒ� (:"cpu0������Babc_b1/Adam_1 = VariableV2()H����ϒ�PbW

Adam/beta1۱��ϒ� (: "cpu0�⑉��BAdam/beta1 = Const()Hӱ��ϒ�Pbm
beta1_power/read���ϒ� (: "cpu0����B(beta1_power/read = Identity(beta1_power)H���ϒ�b ^
beta2_power���ϒ� (: "cpu0������Bbeta2_power = VariableV2()H����ϒ�Pb^
abc_W0���ϒ� (: "�cpu0�˘���Babc_W0 = VariableV2()H����ϒ�Pb�Y

Adam/beta2����ϒ� (: "cpu0�푉��BAdam/beta2 = Const()H߱��ϒ�Pbm
beta2_power/read����ϒ� (: "cpu0������B(beta2_power/read = Identity(beta2_power)H����ϒ�b [
Adam/epsilon����ϒ� (: "cpu0������BAdam/epsilon = Const()H����ϒ�PbX
abc_b0����ϒ� (:"Pcpu0������Babc_b0 = VariableV2()H����ϒ�PbPy
gradients/add_grad/Shape_1����ϒ� (:"cpu0������B$gradients/add_grad/Shape_1 = Const()H����ϒ�b^
abc_W1����ϒ� (: "�cpu0����Babc_W1 = VariableV2()H����ϒ�Pb�{
gradients/Mean_grad/Reshape����ϒ� (:"cpu0������B%gradients/Mean_grad/Reshape = Const()H����ϒ�bg
abc_W0/read����ϒ�(: "�cpu0�˘���Babc_W0/read = Identity(abc_W0)H����ϒ�Pb g
abc_W1/read����ϒ�(: "�cpu0����Babc_W1/read = Identity(abc_W1)H����ϒ�Pb f
abc_b0/read����ϒ� (:"Pcpu0������Babc_b0/read = Identity(abc_b0)H����ϒ�Pb V
abc_b1����ϒ� (:"cpu0�՝���Babc_b1 = VariableV2()H����ϒ�b�
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim����ϒ� (: "cpu0������Bogradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim = Const()H����ϒ�Pbd
abc_b1/read����ϒ� (:"cpu0�՝���Babc_b1/read = Identity(abc_b1)H����ϒ�b 
gradients/add_1_grad/Shape_1����ϒ� (:"cpu0������B&gradients/add_1_grad/Shape_1 = Const()H����ϒ�Pbw
gradients/Mean_grad/Const����ϒ� (:"cpu0������B#gradients/Mean_grad/Const = Const()H����ϒ�bg
global_norm/Const_1����ϒ� (: "cpu0������Bglobal_norm/Const_1 = Const()H����ϒ�bu
clip_by_global_norm/mul/x����ϒ� (: "cpu0�ƒ���B#clip_by_global_norm/mul/x = Const()H����ϒ�Pbf
_arg_Placeholder_0_0����ϒ� (:
"(cpuB_arg_Placeholder_0_0 = _Arg()H����ϒ�b i
_arg_Placeholder_1_0_1����ϒ�(:
"�cpuB_arg_Placeholder_1_0_1 = _Arg()H����ϒ�b h
SqueezeĲ��ϒ� �(�2
cpu:
"(cpuB'Squeeze = Squeeze(_arg_Placeholder_0_0)H����ϒ�b �
MatMul²��ϒ� �(�22
cpu�� �2����ϒ��2���ϒ����������:'%
"��cpu (0��ʈ��B4MatMul = MatMul(_arg_Placeholder_1_0_1, abc_W0/read)H����ϒ�Pb �
gradients/add_grad/Shape޵��ϒ� (2.
cpu 2���ϒ�2����ϒ����������:!"cpu (0������B(gradients/add_grad/Shape = Shape(MatMul)H����ϒ�b �
add۵��ϒ� (22
cpu�� �2���ϒ��2����ϒ����������:'%
"��cpu (0��؈��Badd = Add(MatMul, abc_b0/read)Hֵ��ϒ�Pb �
(gradients/add_grad/BroadcastGradientArgs���ϒ� (2.
cpu 2���ϒ�2����ϒ����������: :#"cpu (0������Bvgradients/add_grad/BroadcastGradientArgs = BroadcastGradientArgs(gradients/add_grad/Shape, gradients/add_grad/Shape_1)H���ϒ�b Y
Relu����ϒ� (:%#
"��cpu 0��؈��BRelu = Relu(add)H���ϒ�Pb �
MatMul_1���ϒ� (2.
cpuxx x2����ϒ�x2����ϒ����������:%#
"xxcpu (0��و��B$MatMul_1 = MatMul(Relu, abc_W1/read)H���ϒ�b �
gradients/add_1_grad/Shape����ϒ� (2.
cpu 2����ϒ�2ڶ��ϒ����������:!"cpu (0��و��B,gradients/add_1_grad/Shape = Shape(MatMul_1)H����ϒ�b �
add_1����ϒ� (2.
cpuxx x2����ϒ�x2����ϒ����������:%#
"xxcpu (0������B"add_1 = Add(MatMul_1, abc_b1/read)H����ϒ�Pb �
*gradients/add_1_grad/BroadcastGradientArgs����ϒ� (2.
cpu 2����ϒ�2Ҷ��ϒ����������: :#"cpu (0��و��B|gradients/add_1_grad/BroadcastGradientArgs = BroadcastGradientArgs(gradients/add_1_grad/Shape, gradients/add_1_grad/Shape_1)H����ϒ�b �
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits����ϒ� (2;
cpuPP (2����ϒ�(2����ϒ�(2����ϒ����������:!
"((cpu (0����:%!
"xxcpu 0������B}SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits = SparseSoftmaxCrossEntropyWithLogits(add_1, Squeeze)H����ϒ�Pb(�
gradients/Mean_grad/Shape����ϒ� (2.
cpu 2����ϒ�2����ϒ����������:!"cpu (0����Bjgradients/Mean_grad/Shape = Shape(SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits)H����ϒ�b �
S_retval_SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_0_0����ϒ� (B�_retval_SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_0_0 = _Retval(SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits)H����ϒ�b �
gradients/Mean_grad/Prod����ϒ� (2.
cpu 2����ϒ�2����ϒ����������: "cpu (0����B�gradients/Mean_grad/Prod = Size(SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits, ^gradients/Mean_grad/Shape)H����ϒ�b �
gradients/Mean_grad/Tile����ϒ� (2.
cpu(( (2����ϒ�(2ƶ��ϒ����������:!
"((cpu (0�����BWgradients/Mean_grad/Tile = Tile(gradients/Mean_grad/Reshape, gradients/Mean_grad/Shape)H����ϒ�Pb �
gradients/Mean_grad/floordiv����ϒ� (: "cpu 0����BAgradients/Mean_grad/floordiv = Snapshot(gradients/Mean_grad/Prod)H����ϒ�b �
gradients/Mean_grad/Cast����ϒ� (2.
cpu 2����ϒ�2����ϒ����������: "cpu (0����B=gradients/Mean_grad/Cast = Cast(gradients/Mean_grad/floordiv)H����ϒ�b �
gradients/Mean_grad/truediv����ϒ� (:
"((cpu 0�����BYgradients/Mean_grad/truediv = RealDiv(gradients/Mean_grad/Tile, gradients/Mean_grad/Cast)H����ϒ�b �
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims����ϒ� (2
cpu:#!
"((cpu 0�����B�gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims = ExpandDims(gradients/Mean_grad/truediv, gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim)H����ϒ�b �
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul¶��ϒ� (:#!
"xxcpu 0������B�gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul = Mul(gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims, SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1)H����ϒ�b �
gradients/add_1_grad/Sum_1̶��ϒ� (2.
cpu 2϶��ϒ�2����ϒ����������:!"cpu (0��Í��B�gradients/add_1_grad/Sum_1 = Sum(gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul, gradients/add_1_grad/BroadcastGradientArgs:1)Hƶ��ϒ�b �
gradients/add_1_grad/Sum˶��ϒ� (:#!
"xxcpu 0������B�gradients/add_1_grad/Sum = Sum(gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul, gradients/add_1_grad/BroadcastGradientArgs)Hȶ��ϒ�Pb �
gradients/add_1_grad/Reshape_1ֶ��ϒ� (:"cpu 0��Í��Bbgradients/add_1_grad/Reshape_1 = Reshape(gradients/add_1_grad/Sum_1, gradients/add_1_grad/Shape_1)HԶ��ϒ�b �
gradients/add_1_grad/Reshape׶��ϒ� (:#!
"xxcpu 0������B\gradients/add_1_grad/Reshape = Reshape(gradients/add_1_grad/Sum, gradients/add_1_grad/Shape)HѶ��ϒ�Pb �
%gradients/add_1_grad/tuple/group_deps޶��ϒ�(Blgradients/add_1_grad/tuple/group_deps = NoOp(^gradients/add_1_grad/Reshape, ^gradients/add_1_grad/Reshape_1)H۶��ϒ�b �
global_norm/L2Loss_3���ϒ� (2.
cpu 2���ϒ�2׷��ϒ����������: "cpu (0��و��Beglobal_norm/L2Loss_3 = L2Loss(gradients/add_1_grad/Reshape_1, ^gradients/add_1_grad/tuple/group_deps)H߶��ϒ�b �
 gradients/MatMul_1_grad/MatMul_1���ϒ� (	22
cpu�� �2���ϒ��2����ϒ����������:'%"��cpu (0�����Bugradients/MatMul_1_grad/MatMul_1 = MatMul(Relu, gradients/add_1_grad/Reshape, ^gradients/add_1_grad/tuple/group_deps)H���ϒ�Pb �
gradients/MatMul_1_grad/MatMul���ϒ� (
22
cpu�� �2����ϒ��2ʷ��ϒ����������:'%
"��cpu (0��Í��Bzgradients/MatMul_1_grad/MatMul = MatMul(gradients/add_1_grad/Reshape, abc_W1/read, ^gradients/add_1_grad/tuple/group_deps)H���ϒ�Pb �
global_norm/L2Loss_2����ϒ� (2.
cpu 2����ϒ�2ַ��ϒ����������: "cpu (0��Í��B`global_norm/L2Loss_2 = L2Loss(gradients/MatMul_1_grad/MatMul_1, ^gradients/MatMul_1_grad/MatMul)H����ϒ�b �
gradients/Relu_grad/ReluGrad����ϒ� (:%#
"��cpu 0��Í��Bpgradients/Relu_grad/ReluGrad = ReluGrad(gradients/MatMul_1_grad/MatMul, Relu, ^gradients/MatMul_1_grad/MatMul_1)H����ϒ�Pb �
gradients/add_grad/Sum����ϒ� (:%#
"��cpu 0��Í��Bdgradients/add_grad/Sum = Sum(gradients/Relu_grad/ReluGrad, gradients/add_grad/BroadcastGradientArgs)H����ϒ�b �
gradients/add_grad/Sum_1����ϒ� (2.
cpuPP P2����ϒ�P2����ϒ����������:!"PPcpu (0��؈��Bhgradients/add_grad/Sum_1 = Sum(gradients/Relu_grad/ReluGrad, gradients/add_grad/BroadcastGradientArgs:1)H����ϒ�Pb �
gradients/add_grad/Reshape����ϒ� (:%#
"��cpu 0��Í��BVgradients/add_grad/Reshape = Reshape(gradients/add_grad/Sum, gradients/add_grad/Shape)H����ϒ�b �
gradients/add_grad/Reshape_1����ϒ� (:"PPcpu 0��؈��B\gradients/add_grad/Reshape_1 = Reshape(gradients/add_grad/Sum_1, gradients/add_grad/Shape_1)H����ϒ�Pb �
global_norm/L2Loss_1����ϒ� (2.
cpu 2����ϒ�2ַ��ϒ����������: "cpu (0��و��BXglobal_norm/L2Loss_1 = L2Loss(gradients/add_grad/Reshape_1, ^gradients/add_grad/Reshape)H����ϒ�b �
gradients/MatMul_grad/MatMul_1ķ��ϒ� (22
cpu�� �2ŷ��ϒ��2����ϒ����������:'%"��cpu (0��Í��Bzgradients/MatMul_grad/MatMul_1 = MatMul(_arg_Placeholder_1_0_1, gradients/add_grad/Reshape, ^gradients/add_grad/Reshape_1)H����ϒ�b �
global_norm/L2Loss̷��ϒ� (2.
cpu 2ͷ��ϒ�2ַ��ϒ����������: "cpu (0��Í��B;global_norm/L2Loss = L2Loss(gradients/MatMul_grad/MatMul_1)Hʷ��ϒ�b �
global_norm/stackз��ϒ� (2.
cpu 2ҷ��ϒ�2ݷ��ϒ����������:!"cpu (0��Í��Bnglobal_norm/stack = Pack(global_norm/L2Loss, global_norm/L2Loss_1, global_norm/L2Loss_2, global_norm/L2Loss_3)HϷ��ϒ�b �
global_norm/Sumٷ��ϒ� (2.
cpu 2ڷ��ϒ�2����ϒ����������: "cpu (0��Í��BCglobal_norm/Sum = Sum(global_norm/stack, gradients/Mean_grad/Const)H׷��ϒ�b �
global_norm/mul޷��ϒ� (: "cpu 0��Í��B;global_norm/mul = Mul(global_norm/Sum, global_norm/Const_1)Hݷ��ϒ�b �
1clip_by_global_norm/truediv/unary_ops_composition���ϒ� (: "cpu 0��Í��BYclip_by_global_norm/truediv/unary_ops_composition = _UnaryOpsComposition(global_norm/mul)H���ϒ�b �
clip_by_global_norm/Minimum���ϒ� (: "cpu 0��Í��Bsclip_by_global_norm/Minimum = Minimum(clip_by_global_norm/truediv/unary_ops_composition, clip_by_global_norm/mul/x)H���ϒ�b �
clip_by_global_norm/mul���ϒ� (: "cpu 0��Í��B?clip_by_global_norm/mul = Snapshot(clip_by_global_norm/Minimum)H���ϒ�b �
clip_by_global_norm/mul_4����ϒ� (:"cpu 0��Í��BXclip_by_global_norm/mul_4 = Mul(gradients/add_1_grad/Reshape_1, clip_by_global_norm/mul)H���ϒ�b �
clip_by_global_norm/mul_1����ϒ� (:%#"��cpu 0��Í��BXclip_by_global_norm/mul_1 = Mul(gradients/MatMul_grad/MatMul_1, clip_by_global_norm/mul)H���ϒ�Pb �
Adam/update_abc_W0/ApplyAdam����ϒ� (
: "�cpu0�˘���B�Adam/update_abc_W0/ApplyAdam = ApplyAdam(abc_W0, abc_W0/Adam, abc_W0/Adam_1, beta1_power/read, beta2_power/read, Adam/learning_rate, Adam/beta1, Adam/beta2, Adam/epsilon, clip_by_global_norm/mul_1)H����ϒ�b �
Adam/update_abc_b1/ApplyAdam����ϒ� (:"cpu0�՝���B�Adam/update_abc_b1/ApplyAdam = ApplyAdam(abc_b1, abc_b1/Adam, abc_b1/Adam_1, beta1_power/read, beta2_power/read, Adam/learning_rate, Adam/beta1, Adam/beta2, Adam/epsilon, clip_by_global_norm/mul_4)H����ϒ�Pb �
clip_by_global_norm/mul_2����ϒ� (:"PPcpu 0��؈��BVclip_by_global_norm/mul_2 = Mul(gradients/add_grad/Reshape_1, clip_by_global_norm/mul)H���ϒ�b �
clip_by_global_norm/mul_3����ϒ� (:%#"��cpu 0�����BZclip_by_global_norm/mul_3 = Mul(gradients/MatMul_1_grad/MatMul_1, clip_by_global_norm/mul)H���ϒ�Pb �
Adam/update_abc_b0/ApplyAdam����ϒ� (:"Pcpu0������B�Adam/update_abc_b0/ApplyAdam = ApplyAdam(abc_b0, abc_b0/Adam, abc_b0/Adam_1, beta1_power/read, beta2_power/read, Adam/learning_rate, Adam/beta1, Adam/beta2, Adam/epsilon, clip_by_global_norm/mul_2)H����ϒ�b �
Adam/update_abc_W1/ApplyAdam����ϒ� (: "�cpu0����B�Adam/update_abc_W1/ApplyAdam = ApplyAdam(abc_W1, abc_W1/Adam, abc_W1/Adam_1, beta1_power/read, beta2_power/read, Adam/learning_rate, Adam/beta1, Adam/beta2, Adam/epsilon, clip_by_global_norm/mul_3)H����ϒ�Pb �

Adam/mul_1����ϒ� (2.
cpu 2����ϒ�2����ϒ����������: "cpu (0��č��B�Adam/mul_1 = Mul(beta2_power/read, Adam/beta2, ^Adam/update_abc_W0/ApplyAdam, ^Adam/update_abc_W1/ApplyAdam, ^Adam/update_abc_b0/ApplyAdam, ^Adam/update_abc_b1/ApplyAdam)H����ϒ�b �
Adam/mul����ϒ� (2.
cpu 2����ϒ�2����ϒ����������: "cpu (0�ΰ���B�Adam/mul = Mul(beta1_power/read, Adam/beta1, ^Adam/update_abc_W0/ApplyAdam, ^Adam/update_abc_W1/ApplyAdam, ^Adam/update_abc_b0/ApplyAdam, ^Adam/update_abc_b1/ApplyAdam)H����ϒ�Pb q
Adam/Assign_1����ϒ� (: "cpu0������B/Adam/Assign_1 = Assign(beta2_power, Adam/mul_1)H����ϒ�b m
Adam/Assign����ϒ� (: "cpu0����B+Adam/Assign = Assign(beta1_power, Adam/mul)H����ϒ�b G
Adam����ϒ�(B)Adam = NoOp(^Adam/Assign, ^Adam/Assign_1)H����ϒ�b ~0��