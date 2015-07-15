% set up a convolutional auto-encoder with random initialization
% input channels | output channels | kernel size | pool size | noise
% weights of encoder and decoder networks are tied
function [ cae ] = cae_setup(ic,oc,ks,ps,noise)
    cae.ic = ic; % input channels
    cae.oc = oc; % output channels
    cae.ks = ks; % kernel size
    cae.ps = ps; % pool size
    cae.noise = noise;
    
    cae.b = zeros(cae.oc,1);
    cae.c = zeros(cae.ic,1);
    cae.w = (rand([cae.ks cae.ks cae.ic cae.oc])-0.5)*2*30/(cae.oc*cae.ks^2);
    cae.w_tilde = flip(flip(cae.w,1),2);
end

