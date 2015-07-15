% visualize the first layer kernels of CAE,
% and randomly selected reconstruction results aside the original input.
function cae_vis(cae, x)
    figure,plot(cae.L);
    % plot first layer kernels
    w = zeros(cae.ks,(cae.ks+2)*cae.oc,cae.ic);
    for oc = 1:cae.oc
        w(1:cae.ks,(cae.ks+2)*(oc-1)+2:(cae.ks+2)*oc-1,:) = cae.w(:,:,:,oc);
    end
%     figure,imshow(imresize(w,10,'nearest'));
    figure,imshow(imresize(w,10,'nearest'),[min(w(:)) max(w(:))]);
    
    % plot randomly selected reconstruction results
    n = 10;
    sample_id = randi(size(x,4),1,n^2);
    x = x(:,:,:,sample_id);
    opts.alpha = 0;
    opts.numepochs = 1;
    opts.batchsize = n^2;
    opts.shuffle=0;
    cae_tmp = cae_train(cae, x, opts);
    
    input = zeros(n*(size(x,1)+2),n*(size(x,2)+2),size(x,3))+0.5;
    recon = zeros(n*(size(x,1)+2),n*(size(x,2)+2),size(x,3))+0.5;
   
    for i = 1:n
        for j=1:n
            input((i-1)*(size(x,1)+2)+2:i*(size(x,1)+2)-1,(j-1)*(size(x,1)+2)+2:j*(size(x,1)+2)-1,:)=x(:,:,:,(i-1)*n+j);
            recon((i-1)*(size(x,1)+2)+2:i*(size(x,1)+2)-1,(j-1)*(size(x,1)+2)+2:j*(size(x,1)+2)-1,:)=cae_tmp.o(:,:,:,(i-1)*n+j);
        end        
    end
    figure,
    subplot(1,2,1),imshow(input);
    subplot(1,2,2),imshow(recon);
end
