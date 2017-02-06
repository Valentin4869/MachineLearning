function [frames, d, f_N]=getFrames_Dxyz(path)
%
% Return the per-frame joint displacement for all frames
        
    skeleton=loadbvh(path);
    
    f_N=size(skeleton(1).Dxyz,2);
    elements_n=size(skeleton,2);
    d=elements_n*3; %dimensionality of an individual frame + 3 for root displ
    frames=zeros(d,f_N);
    
    
    for j=1:elements_n
      %  if ~isempty(joints(j).rxyz)
          frames(((j-1)*3+1):(j*3),:)=skeleton(j).Dxyz(:,1:end);
      %  else
     %     elements_n=elements_n-1; % subtract empty nodes
     %   end
     
    end
    
%downsample test
frames=frames(:,1:4:end);
f_N=size(frames,2);
end