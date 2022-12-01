def make_debug_point(port=9993):
    import ptvsd

    ptvsd.enable_attach(address=("localhost", port))
    print(f"Waiting for attach remote debug, port: {port}")
    ptvsd.wait_for_attach()
