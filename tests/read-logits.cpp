#include "llama.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) try {
    if (argc < 5 || argc > 260) {
        throw std::runtime_error("usage: read-logits MODEL THREADS KV_TYPE TOKEN...");
    }
    auto model_params         = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], model_params),
                                                                    llama_model_free);
    if (!model) {
        return 1;
    }

    auto params      = llama_context_default_params();
    params.n_ctx     = 256;
    params.n_threads = params.n_threads_batch = std::stoi(argv[2]);
    if (std::string(argv[3]) == "q8_0") {
        params.type_k = params.type_v = GGML_TYPE_Q8_0;
        params.flash_attn_type        = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), params), llama_free);
    if (!ctx) {
        return 1;
    }

    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),
                                                                            llama_batch_ext_free);
    for (int i = 4; i < argc; ++i) {
        const int32_t   idx = llama_batch_ext_add_token(batch.get(), 0, std::stoi(argv[i]));
        const llama_pos pos = i - 4;
        if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, &pos) ||
            !llama_batch_ext_set_output_logits(batch.get(), idx, true)) {
            throw std::runtime_error("cannot add token to batch");
        }
    }
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        return 1;
    }

    const size_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    for (int i = 0; i < argc - 4; ++i) {
        const float * logits = llama_get_logits_ith(ctx.get(), i);
        if (!logits || std::fwrite(logits, sizeof(float), vocab, stdout) != vocab) {
            return 1;
        }
    }
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
