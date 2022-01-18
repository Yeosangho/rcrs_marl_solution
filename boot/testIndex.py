import torch
a = torch.Tensor([1,2,3,4]).long()
b = torch.Tensor([2,3,1,0]).long()

print(a[b[:]])

a = { "c" : [1,2], "d" : [3,4], "a" : [4,5], "b" : [5,6]}
x = {}
for e in a.keys() :
    if (e == "a" or e == "d"):
        x[e] = a[e]
print(x)       

x = sorted(x.items())

x = dict(x)
a["a"][0] = 10
print(x)       



#a = torch.Tensor([1,2,3,1,2]).long()
#b = torch.Tensor([2,1,2,3,1]).long()
#c = torch.Tensor([1,5,3,4,2]).long()
#d = torch.zeros(5,5).float()
#
#x = torch.add(d[a[:], b[:]], c[:])
#d[a[:], b[:]] = c[:] / 2.
#a = torch.Tensor([1,]).long()
#b = torch.Tensor([2,]).long()
#c = torch.Tensor([111,]).long()
#d[a[:], b[:]] = c[:] / 2.
#print(d[a[:], b[:]])
#print(d)