"""Transparent major-operation accounting; unsupported costs remain explicit."""


def count_flops(model, length=144, components=15, width=32, state=64, grids=8, method='FKANMamba'):
    from .protocol import VARIANTS
    if method not in ('FKANMamba',*VARIANTS):
        return dict(major_macs=None,major_linear_flops=None,complete_flops=False,
                    restriction='Unsupported baseline: supply validated per-model operation accounting.')
    variant=getattr(model,'variant','FKANMamba')
    classes=model.classifier.units
    mlp=2*width*128
    basis=width*grids*width
    projection=width*width if variant=='linear_qkv' else (mlp if variant=='mlp_qkv' else basis)
    parts={'token_projections':2*length*components*width,
           'feb_projections':0 if variant=='without_feb' else 2*(mlp if variant=='mlp_gates' else basis),
           'qkv_projections':0 if variant=='without_fssa' else 2*3*length*projection,
           'attention_qk_and_av':0 if variant=='without_fssa' else 2*2*length*length*width,
           'attention_output_projections':0 if variant=='without_fssa' else 2*length*width*width,
           'recurrent_dense_maps':length*(state*state+2*width*state),
           'classifier':state*classes}
    excluded=['bias additions','LayerNorm','softmax','dropout','sigmoid',
              'RBF subtraction/division/squaring/exponential','elementwise gating',
              'state summation','memory movement and padding']
    if variant=='mean_pool':
        parts['recurrent_dense_maps']=0
        excluded+=['mean-pooling reductions and division']
    if variant=='selective_mamba':
        inner,rank=128,4
        parts['recurrent_dense_maps']=0
        parts['selective_in_projection']=length*64*2*inner
        parts['selective_depthwise_convolution']=length*inner*4
        parts['selective_parameter_projection']=length*inner*(rank+2*64)
        parts['selective_delta_projection']=length*rank*inner
        parts['selective_output_projection']=length*inner*64
        excluded+=['selective state scan elementwise arithmetic','SiLU','softplus']
    return dict(major_macs=sum(parts.values()), major_linear_flops=2*sum(parts.values()),
                macs_by_operation=parts, batch_size=1,
                convention='One multiply-accumulate = two FLOPs. Dense, convolution, QK and AV only.',
                complete_flops=False, excluded_operations=excluded,
                restriction='Do not present major_linear_flops as complete model FLOPs or compare it with full profiler totals.')
