function plotGrid(R,G,B,t)
%PLOTGRID Summary of this function goes here
%   Detailed explanation goes here

for i = 1:100
    R_i=R(i,:);
    G_i=G(i,:);
    B_i=B(i,:);
    A(:,:,1,i)=reshape(R_i,32,32)';
    A(:,:,2,i)=reshape(G_i,32,32)';
    A(:,:,3,i)=reshape(B_i,32,32)';
end

figure
thumbnails = A(:,:,:,1:100);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails)
title(t);

end

