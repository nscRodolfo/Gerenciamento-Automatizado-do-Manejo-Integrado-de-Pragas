package my.aplication.manejointeligentedepragas.model;

public class FuncionarioModel {
    private int Cod_funcionario;
    private int fk_Cod_Usuario;



    public FuncionarioModel(int cod_funcionario, int fk_Cod_Usuario) {
        this.Cod_funcionario = cod_funcionario;
        this.fk_Cod_Usuario = fk_Cod_Usuario;
    }

    public int getCod_funcionario() {
        return Cod_funcionario;
    }
    public void setCod_funcionario(int cod_funcionario) {
        this.Cod_funcionario = cod_funcionario;
    }


    public int getFk_Cod_Usuario() {
        return fk_Cod_Usuario;
    }

    public void setFk_Cod_Usuario(int fk_Cod_Usuario) {
        this.fk_Cod_Usuario = fk_Cod_Usuario;
    }
}
