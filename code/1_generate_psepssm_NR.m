clear all
clc
lamdashu=10;
%Reading protein sequence of PSSM
WEISHU=26;
name=textread('.\1_original_data\NR\nr_target_name.txt','%s')    %Read protein name
for i=1:26 % #(nr_target_name)=26
    nnn=name(i);
    nnn1=strcat('.\1_original_data\NR\Nuclear receptor-PSSM\',nnn,'.pssm');  %Read PSSM file   
    nnn1=char(nnn1);
    fid{i}=importdata(nnn1); 
end
%All protein sequences normalized
c=cell(WEISHU,1);
for t=1:WEISHU
    clear shu d
shu=fid{t}.data; %
%Know the quantity of each protein, the extracted matrix, pay attention to the order of the protein
[M,N]=size(shu);
shuju=shu(1:M-5,1:20);
d=[];
%Normalized
for i=1:M-5
   for j=1:20
       d(i,j)=1/(1+exp(-shuju(i,j)));
   end
end
c{t}=d(:,:);
end
%Generate PSSM-AAC
for i=1:WEISHU
[MM,NN]=size(c{i});
 for  j=1:20
   x(i,j)=sum(c{i}(:,j))/MM;
 end
end
%After PsePSSM 20*lamda
xx=[];
sheta=[];
shetaxin=[];
for lamda=1:lamdashu;
for t=1:WEISHU
  [MM,NN]=size(c{t});
  clear xx
   for  j=1:20
      for i=1:MM-lamda
       xx(i,j)=(c{t}(i,j)-c{t}(i+lamda,j))^2;
      end
      sheta(t,j)=sum(xx(1:MM-lamda,j))/(MM-lamda);
   end
end
shetaxin=[shetaxin,sheta];
end
psepssm=[x,shetaxin];
xlswrite('.\1_original_data\NR\nr_psepssm_drop4.xlsx',psepssm)