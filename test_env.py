import src.rl_env.env_ls_with_short_reward as env
import src.utils.utils as utils

urls = './data/STOCKS_GOOGL.csv'

stock_info = utils.load_data(urls)

env = env.MyEnv(stock_info)

default_option = {
    'window_size': 20,
    'start_index': 0,
    'end_index': 100,
    'commission': 0.01,
    'selling_tax': 0.01,
}


env.reset()


while not env._done:
    user_input = input("Enter your action [ 0 : short , 1 : long ] : ")
    if (user_input == '0') or (user_input == '1'):
        print("done : ", env._done, "")
        env.step(int(user_input))
        env.print_history()
    else:
        env.print_info()

env.print_info()
print("Over")
env.print_history()