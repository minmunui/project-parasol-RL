import src.env.env_bs as env
import src.utils.utils as utils

urls = './data/STOCKS_GOOGL.csv'

stock_info = utils.load_data(urls)

env = env.MyEnv(stock_info)

print(env.reset())
print(env.df)

print(env.init_balance)

while True:
    while env._done is False:
        print("env.price", env.price)
        user_input = input("Enter your action \"\": ")
        env.step(int(user_input))
        env.print_current_state()

    env.reset()
