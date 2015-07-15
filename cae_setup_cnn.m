% use cae to initialize cnn of the same structure
% only the first layer by far, to be improved
function [cnn] = cae_setup_cnn(cae, cnn, train_x, train_y)
    cnn = cnnsetup(cnn,train_x,train_y);
    for j = 1 : cae.oc
        for i = 1 : cae.ic
            cnn.layers{2}.k{i}{j} = cae.w(:,:,i,j);
        end
        cnn.layers{2}.b{j} = cae.b(j);
    end
end
