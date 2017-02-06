function frames=rxyz2dxyz(rots,parents, offsets, fps,fskip,c)

if nargin==2
    fskip=1;
    c='b'
elseif nargin==1
        
    fps=60; 
    fskip=1;
    c='b'
end

%%rots=pred_seed.items;

f_n=size(rots,2)
t_root=rots(1:3,:);
rots=rots(4:end,:); % exclude root translations from rotation data
frames=zeros(size(rots));
n=(size(rots,1))/3;

for f=1:f_n


%forgot to take into account rots contains root translation

%initialize root 
% 
%  th_x=joints(1).rxyz(1,f);
%  th_y=joints(1).rxyz(2,f);
%  th_z=joints(1).rxyz(3,f);  
%   

t_mat=zeros(4,4,size(rots,1)/3); % store transform mat of each joint

t_mat(:,:,1)=[(rotz(rots(3,f))*rotx(rots(1,f))*roty(rots(2,f))) t_root(1:3,f);0 0 0 1];

frames(1:3,f)=t_mat([1 2 3], 4, 1);



 for i=2:n
  
% 
% if isempty(joints(i).rxyz)
%   th_x=0;
%   th_y=0;
%   th_z=0;
%   
% else
%   th_x=joints(i).rxyz(1,f);
%   th_y=joints(i).rxyz(2,f);
%   th_z=joints(i).rxyz(3,f);  
% end
%   
%   th_xp=double(joints(j).rxyz(1,f))*c;
%   th_yp=double(joints(j).rxyz(2,f))*c;
%   th_zp=double(joints(j).rxyz(3,f))*c;
  
  
tp=t_mat(:,:,parents(i));
rz=rotz(rots(3+(i-1)*3,f));
rx=rotx(rots(1+(i-1)*3,f));
ry=roty(rots(2+(i-1)*3,f));
  %t_mat(:,:,i)=tmtx(:,:,j)*[(rotz(th_z)*rotx(th_x)*roty(th_y)) joints(i).offset;0 0 0 1];
 t_mat(:,:,i)=tp*[rz*rx*ry offsets(i,:)';0 0 0 1];
  %t_mat(:,:,i)=t_mat(:,:,parents(i))*[(rotz(rots(3+(i-1)*3,f))*rotx(rots(1+(i-1)*3,f))*roty(rots(2+(i-1)*3,f))) offsets(i,:)';0 0 0 1];
  %ptrans=[(rotz(th_zp)*rotx(th_xp)*roty(th_yp)) joints(j).Dxyz(1:3,f);0 0 0 1];
  %trans=[(rotz(th_z)*rotx(th_x)*roty(th_y)) joints(i).offset;0 0 0 1]*[frame((1+(j-1)*3):(3*j)); 1];
  
  
  
  
     frames((1+(i-1)*3):(3*i),f)=t_mat([1 2 3], 4,i);
%      
%      if strcmp(joints(i).name,'Head') ||strcmp(joints(i).name,'RightHand')||strcmp(joints(i).name,'LeftHand')||strcmp(joints(i).name,'RightArm')||strcmp(joints(i).name,'LeftArm')
%      text(trans(1, 4),trans(2, 4),trans(3, 4),joints(i).name)
%      end
 end
 
           % hold on
            
          %   plot3(frames(1:3:end,f),frames(2:3:end,f),frames(3:3:end,f),'.');


           %    view(0,90);
              
            %    pause(0.2)
end

end