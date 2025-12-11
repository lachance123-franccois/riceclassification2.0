def test_forward_pass(model, dummy_data):
    x, y = dummy_data
    output = model(x)
    assert output.shape == y.shape
    assert (output >=0).all() and (output <=1).all()  # Sigmoid output
