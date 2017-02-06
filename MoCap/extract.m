function ex=extract(t_set,i)

ex= reshape(t_set(i,:,:),[size(t_set,2), size(t_set,3)]);

end