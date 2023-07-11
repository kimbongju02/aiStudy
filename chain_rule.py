

# 합성함수 미분

class Mul():
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        result = x * y
        return result
    
    def backward(self, dresult) :
        dx = dresult * self.y 
        dy = dresult * self.x
        return dx, dy 
    
class Add():
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        result = x + y
        return result
    
    def backward(self, dresult):
        dx = dresult * 1
        dy = dresult * 1
        return dx, dy
    
a, b, c = -1, 3, 4
x = Add()
y = Add()
f = Mul()   

x_result = x.forward(a, b)
y_result = y.forward(b, c)

print(x_result)
print(y_result)
print(f.forward(x_result, y_result))

dresult = 1
dx_mul, dy_mul = f.backward(dresult)

da_add, db_add_1 = x.backward(dx_mul)
db_add_2, dc_add = y.backward(dy_mul)

print(dx_mul, dy_mul)
print(da_add)
print(db_add_1 + db_add_2)
print(dc_add)