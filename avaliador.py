import os
import subprocess
import statistics
import shutil
import time

# --- Bloco de Configuração ---

# Parâmetros que serão passados para os programas C
PARAMS = {
    "DATASET_FILE": "dataset.txt",
    "M_POINTS": 1000000,
    "D_DIMENSIONS": 10,
    "K_CLUSTERS": 100,
    "I_ITERATIONS": 50,
}

# Número de vezes que cada executável será rodado para tirar a média
NUM_RUNS = 30

# Detecta o número de núcleos de CPU disponíveis para usar nos testes paralelos
CPU_CORES = os.cpu_count() or 4 # Usa 4 como padrão se a detecção falhar

# Lista de executáveis a serem testados
EXECUTABLES = [
    {"name": "Sequencial", "source": "kmeans_sequencial.c", "output": "kmeans_sequencial", "type": "serial", "compile_cmd": "gcc -o kmeans_sequencial kmeans_sequencial.c -O3"},
    {"name": "OpenMP", "source": "kmeans_openmp.c", "output": "kmeans_openmp", "type": "omp", "compile_cmd": "gcc -o kmeans_openmp kmeans_openmp.c -fopenmp -O3"},
    {"name": "Pthreads", "source": "kmeans_pthreads.c", "output": "kmeans_pthreads", "type": "serial", "compile_cmd": "gcc -o kmeans_pthreads kmeans_pthreads.c -lpthread -O3"},
    {"name": "MPI", "source": "kmeans_mpi.c", "output": "kmeans_mpi", "type": "mpi", "compile_cmd": "mpicc -o kmeans_mpi kmeans_mpi.c -O3"}
]

# --- Cores para o Terminal ---
class C:
    HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'; END = '\033[0m'; BOLD = '\033[1m'

# --- Funções do Avaliador ---

def check_dependencies():
    """Verifica se os compiladores necessários estão instalados."""
    print(f"{C.HEADER}--- Verificando Dependências ---{C.END}")
    if not shutil.which("gcc"): print(f"{C.RED}Erro: Compilador 'gcc' não encontrado.{C.END}"); exit(1)
    if not shutil.which("mpicc"): print(f"{C.RED}Erro: Compilador 'mpicc' não encontrado.{C.END}"); exit(1)
    print(f"{C.GREEN}Compiladores 'gcc' e 'mpicc' encontrados.{C.END}\n")

def compile_sources():
    """Compila todos os arquivos fonte C definidos na lista EXECUTABLES."""
    print(f"{C.HEADER}--- Compilando Códigos Fonte ---{C.END}")
    for exe in EXECUTABLES:
        print(f"Compilando {C.YELLOW}{exe['name']}{C.END} ({exe['source']})... ", end='', flush=True)
        try:
            subprocess.run(exe['compile_cmd'], shell=True, check=True, capture_output=True, text=True)
            print(f"{C.GREEN}OK{C.END}")
        except subprocess.CalledProcessError as e:
            print(f"{C.RED}FALHOU{C.END}\n{e.stderr}"); exit(1)
    print()

def get_golden_checksum(args):
    """Executa a versão sequencial uma vez para obter o checksum de referência."""
    print(f"{C.HEADER}--- Obtendo Checksum de Referência ---{C.END}")
    try:
        seq_exe = next(e for e in EXECUTABLES if e['name'] == 'Sequencial')
        # Passa os argumentos para a chamada
        cmd = [f"./{seq_exe['output']}"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        _, checksum_str = result.stdout.strip().split('\n')
        golden_checksum = int(checksum_str)
        print(f"{C.GREEN}Checksum de referência obtido: {golden_checksum}{C.END}\n")
        return golden_checksum
    except (subprocess.CalledProcessError, StopIteration, ValueError, IndexError) as e:
        print(f"{C.RED}Erro ao obter o checksum de referência: {e}{C.END}"); exit(1)

def run_benchmark(golden_checksum, args):
    """Executa cada programa, coleta os tempos e verifica os checksums."""
    results = []
    print(f"{C.HEADER}--- Iniciando Benchmark (Hardware: {CPU_CORES} núcleos) ---{C.END}")
    
    for exe in EXECUTABLES:
        print(f"{C.BLUE}Avaliando: {C.BOLD}{exe['name']}{C.END}")
        times, correct_runs = [], 0
        
        # Passa os argumentos para o comando base
        base_cmd = [f"./{exe['output']}"] + args
        run_env = os.environ.copy()
        if exe['type'] == 'omp': run_env['OMP_NUM_THREADS'] = str(CPU_CORES)
        elif exe['type'] == 'mpi': base_cmd = ["mpirun", "-np", str(CPU_CORES)] + base_cmd
            
        for i in range(NUM_RUNS):
            print(f"  Execução {i + 1}/{NUM_RUNS}... ", end='', flush=True)
            try:
                result = subprocess.run(base_cmd, env=run_env, capture_output=True, text=True, check=True)
                time_str, checksum_str = result.stdout.strip().split('\n')
                duration, checksum = float(time_str), int(checksum_str)
                times.append(duration)
                if checksum == golden_checksum: correct_runs += 1
                print(f"Tempo: {duration:.4f}s, Checksum: {'OK' if checksum == golden_checksum else 'FALHOU'}")
            except (subprocess.CalledProcessError, ValueError, IndexError):
                print(f"{C.RED}FALHOU (erro na execução ou saída inválida){C.END}")

        avg_time = statistics.mean(times) if times else 0.0
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
        
        results.append({"name": exe['name'], "avg_time": avg_time, "stdev": stdev_time, "correct_runs": correct_runs})
        print(f"{C.GREEN}  Resultado {exe['name']}: Média = {avg_time:.4f}s, Corretude = {correct_runs}/{NUM_RUNS}{C.END}\n")
        
    return results

def print_summary(results, golden_checksum):
    """Imprime uma tabela com o resumo dos resultados."""
    print(f"{C.HEADER}--- Resumo dos Resultados (Checksum de Referência: {golden_checksum}) ---{C.END}")
    
    try:
        sequential_time = next(r['avg_time'] for r in results if r['name'] == 'Sequencial')
    except StopIteration:
        print(f"{C.RED}Erro: Versão 'Sequencial' não encontrada.{C.END}"); return

    print(f"{C.BOLD}{'Versão':<15} | {'Tempo Médio (s)':<18} | {'Speedup':<12} | {'Corretude':<15}{C.END}")
    print("-" * 65)
    
    for res in results:
        name, avg_time = res['name'], res['avg_time']
        speedup = sequential_time / avg_time if avg_time > 0 else 0
        correctness = f"({res['correct_runs']}/{NUM_RUNS})"
        status_color = C.GREEN if res['correct_runs'] == NUM_RUNS else C.RED
        print(f"{name:<15} | {avg_time:<18.4f} | {f'{speedup:.2f}x' if name != 'Sequencial' else '1.00x':<12} | {status_color}{correctness:<15}{C.END}")
        
    print("-" * 65)

# --- Ponto de Entrada Principal ---

if __name__ == "__main__":
    check_dependencies()
    compile_sources()
    
    # Monta a lista de argumentos a partir do dicionário PARAMS
    main_args = [
        PARAMS["DATASET_FILE"],
        str(PARAMS["M_POINTS"]),
        str(PARAMS["D_DIMENSIONS"]),
        str(PARAMS["K_CLUSTERS"]),
        str(PARAMS["I_ITERATIONS"])
    ]
    
    # Passa os argumentos para as funções
    golden_checksum_val = get_golden_checksum(main_args)
    benchmark_results = run_benchmark(golden_checksum_val, main_args)
    print_summary(benchmark_results, golden_checksum_val)