function animate_bvh(path,fps,fskip)

if nargin==2
    fskip=1;
    
elseif nargin==1
        
    fps=60; 
    fskip=1;
    
end

   
    skeleton=loadbvh(path);
  
    frames_n=size(skeleton(1).Dxyz,2);
    elements_n=size(skeleton,2);
    frames_matrix=zeros(3*elements_n,frames_n);
    parents=zeros(elements_n,1);

    for j=1:elements_n
          if ~isempty(skeleton(j).rxyz)
             frames_matrix(((j-1)*3+1):(j*3),:)=skeleton(j).Dxyz;
             parents(j,1)=skeleton(j).parent;
          end
    end
    
    animate_frames(frames_matrix,fps,fskip)
    
end