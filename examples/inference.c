#include "darknet.h"

float * get_dummy_input(int length){
    float * ret = (float*)malloc(sizeof(float) * length);
    int i;
    for (i=0;i<length;i++){
        ret[i] = (float)((length % 10) - 5) * 0.0001;
    }
    return ret;
}

void test_inference(char *cfgfile, char *weightfile, int layer_index)
{
    int i;
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    int net_w = net->w, net_h = net->h;
    printf("net_h=%d, net_w=%d\n", net_h, net_w);

    float * dummy_input = get_dummy_input(net_w * net_h);
    network_predict(net, dummy_input);

    printf("inspect layer index=%d\n", layer_index);
    layer l = net->layers[layer_index];
    
    printf("layer nchw: (%d, %d, %d, %d)\n", l.batch, l.n, l.h, l.w);
    float * out = l.output;

    int dim = l.n * l.h * l.w;
    if (dim <= 100) {
        for (i=0;i<dim;i++){
            printf("%.5f, ", out[i]);
        }
        printf("\n");
    } else {
        printf("first 25 value:\n");
        for (i=0;i<25;i++){
            printf("%.5f, ", out[i]);
        }
        printf("\nlast 25 value:\n");
        for (i=0;i<25;i++){
            printf("%.5f, ", out[dim - 25 + i]);
        }
    }
}
