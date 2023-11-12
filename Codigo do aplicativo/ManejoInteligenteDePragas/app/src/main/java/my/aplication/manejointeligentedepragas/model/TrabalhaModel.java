package my.aplication.manejointeligentedepragas.model;

public class TrabalhaModel {
    private int fk_Cod_Propriedade;
    private int fk_Cod_Funcionario;

    public TrabalhaModel(int fk_Cod_Propriedade, int fk_Cod_Funcionario) {
        this.fk_Cod_Propriedade = fk_Cod_Propriedade;
        this.fk_Cod_Funcionario = fk_Cod_Funcionario;
    }

    public int getFk_Cod_Propriedade() {
        return fk_Cod_Propriedade;
    }

    public void setFk_Cod_Propriedade(int fk_Cod_Propriedade) {
        this.fk_Cod_Propriedade = fk_Cod_Propriedade;
    }

    public int getFk_Cod_Funcionario() {
        return fk_Cod_Funcionario;
    }

    public void setFk_Cod_Funcionario(int fk_Cod_Funcionario) {
        this.fk_Cod_Funcionario = fk_Cod_Funcionario;
    }
}
